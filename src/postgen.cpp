// postgen — blog post generator for zty2004.github.io
//
// Turns a *.md / *.tex / *.pdf file (or a photo directory) into a Jekyll post
// named "<date>-<Title-Slug>.md", extracting the title automatically when
// possible. Embedded local images are copied into images/<date>-<slug>/ and
// compressed with macOS `sips` (copies only — source files are never touched).
//
// Build:  make            (see src/Makefile, requires -std=c++20)
// Usage:  postgen <file.md|file.tex|file.pdf> [options]
//         postgen --photos <dir> [options]
// Options:
//   --title "..."    override the post title
//   --tags a,b,c     comma-separated tags
//   --date YYYY-M-D  override the date prefix (default: today)
//   --math           force `math: true` in front matter
//   --force          overwrite an existing post
//   --dry-run        print what would happen without writing anything
//   --publish        git add + commit + push the generated files

#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using std::string;

// ---------------------------------------------------------------- options --

struct Options {
    fs::path input;
    bool photosMode = false;
    string title;
    string tags;
    string date;
    bool math = false;
    bool force = false;
    bool dryRun = false;
    bool publish = false;
};

[[noreturn]] static void usage(int code) {
    std::cerr <<
        "usage: postgen <file.md|file.tex|file.pdf> [options]\n"
        "       postgen --photos <dir> [options]\n"
        "options: --title \"...\" --tags a,b --date YYYY-M-D --math --force --dry-run --publish\n";
    std::exit(code);
}

static Options parseArgs(int argc, char** argv) {
    Options o;
    std::vector<string> args(argv + 1, argv + argc);
    for (size_t i = 0; i < args.size(); ++i) {
        const string& a = args[i];
        auto next = [&](const char* flag) -> string {
            if (++i >= args.size()) {
                std::cerr << "postgen: " << flag << " needs a value\n";
                std::exit(2);
            }
            return args[i];
        };
        if (a == "--photos")       { o.photosMode = true; o.input = next("--photos"); }
        else if (a == "--title")   o.title = next("--title");
        else if (a == "--tags")    o.tags = next("--tags");
        else if (a == "--date")    o.date = next("--date");
        else if (a == "--math")    o.math = true;
        else if (a == "--force")   o.force = true;
        else if (a == "--dry-run") o.dryRun = true;
        else if (a == "--publish") o.publish = true;
        else if (a == "-h" || a == "--help") usage(0);
        else if (!a.empty() && a[0] == '-') {
            std::cerr << "postgen: unknown option " << a << "\n";
            usage(2);
        }
        else o.input = a;
    }
    if (o.input.empty()) usage(2);
    return o;
}

// ---------------------------------------------------------------- helpers --

static string shellQuote(const string& s) {
    string out = "'";
    for (char c : s) out += (c == '\'') ? string("'\\''") : string(1, c);
    return out + "'";
}

static string runCapture(const string& cmd) {
    std::array<char, 4096> buf{};
    string out;
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) return out;
    while (fgets(buf.data(), buf.size(), p)) out += buf.data();
    pclose(p);
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
    return out;
}

static string todayDate() {
    std::time_t t = std::time(nullptr);
    std::tm* lt = std::localtime(&t);
    // no leading zeros, matching existing posts like 2024-1-28-...
    return std::to_string(lt->tm_year + 1900) + "-" +
           std::to_string(lt->tm_mon + 1) + "-" +
           std::to_string(lt->tm_mday);
}

static string trim(const string& s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

static string toLower(string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

// spaces/punctuation -> '-', keep ASCII alnum and multi-byte (CJK) chars
static string slugify(const string& title) {
    string out;
    for (unsigned char c : title) {
        if (std::isalnum(c) || c >= 0x80) out += (char)c;
        else if (!out.empty() && out.back() != '-') out += '-';
    }
    while (!out.empty() && out.back() == '-') out.pop_back();
    return out.empty() ? "Untitled" : out;
}

static string readFile(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static fs::path findRepoRoot() {
    for (fs::path d = fs::current_path(); ; d = d.parent_path()) {
        if (fs::exists(d / "_config.yml")) return d;
        if (d == d.root_path()) break;
    }
    std::cerr << "postgen: run inside the blog repository (_config.yml not found)\n";
    std::exit(1);
}

static fs::path expandUser(string p) {
    if (!p.empty() && p[0] == '~') {
        const char* home = std::getenv("HOME");
        if (home) p = string(home) + p.substr(1);
    }
    return p;
}

// ------------------------------------------------------- title extraction --

// returns {body-without-front-matter, title-from-front-matter}
static std::pair<string, string> stripFrontMatter(const string& md) {
    std::regex fmRe(R"(^---\s*\n([\s\S]*?)\n---\s*\n)");
    std::smatch m;
    if (!std::regex_search(md, m, fmRe)) return {md, ""};
    string fm = m[1].str(), title;
    std::smatch tm;
    if (std::regex_search(fm, tm, std::regex(R"(^title:\s*(.+)$)", std::regex::multiline)))
        title = trim(tm[1].str());
    return {md.substr(m[0].length()), title};
}

static string extractH1(string& body) {
    std::regex h1Re(R"(^#\s+(.+)\s*$)", std::regex::multiline);
    std::smatch m;
    if (!std::regex_search(body, m, h1Re)) return "";
    string title = trim(m[1].str());
    body = m.prefix().str() + m.suffix().str();  // drop the H1 line
    return title;
}

// If the body still contains fence-outside H1s, shift ALL headings down one
// level (h1->h2, h2->h3, ..., capped at h6) so parent/child structure is kept
// and the right-side TOC tree — which roots at h2 — captures every section.
static string demoteExtraH1(const string& body) {
    // pass 1: is there any h1 outside code fences?
    {
        std::istringstream in(body);
        string line;
        bool fence = false, hasH1 = false;
        while (std::getline(in, line)) {
            string t = trim(line);
            if (t.rfind("```", 0) == 0 || t.rfind("~~~", 0) == 0) fence = !fence;
            else if (!fence && line.rfind("# ", 0) == 0) hasH1 = true;
        }
        if (!hasH1) return body;
    }
    // pass 2: shift every heading down one level
    std::istringstream in(body);
    std::ostringstream out;
    string line;
    bool fence = false, first = true;
    while (std::getline(in, line)) {
        if (!first) out << "\n";
        first = false;
        string t = trim(line);
        if (t.rfind("```", 0) == 0 || t.rfind("~~~", 0) == 0) fence = !fence;
        size_t hashes = 0;
        while (!fence && hashes < line.size() && line[hashes] == '#') ++hashes;
        if (!fence && hashes >= 1 && hashes < 6 && hashes < line.size() && line[hashes] == ' ')
            out << "#" << line;
        else
            out << line;
    }
    return out.str();
}

static string texTitle(const string& tex) {
    std::smatch m;
    if (std::regex_search(tex, m, std::regex(R"(\\title\{([^}]*)\})")))
        return trim(m[1].str());
    return "";
}

static string pdfTitle(const fs::path& pdf) {
    string t = runCapture("mdls -name kMDItemTitle -raw " + shellQuote(pdf.string()) + " 2>/dev/null");
    if (t == "(null)") t.clear();
    return trim(t);
}

// ------------------------------------------------------ image compression --

static void compressImage(const fs::path& img, bool dryRun) {
    string ext = toLower(img.extension().string());
    if (ext != ".jpg" && ext != ".jpeg" && ext != ".png") return;

    auto before = fs::file_size(img);
    if (before < 300 * 1024) {
        std::cout << "  [keep]  " << img.filename().string()
                  << " (" << before / 1024 << " KB, small enough)\n";
        return;
    }
    string q = shellQuote(img.string());
    int width = std::atoi(runCapture("sips -g pixelWidth " + q + " | awk '/pixelWidth/ {print $2}'").c_str());

    string cmd;
    if (ext == ".png") {
        if (width <= 1600) {
            std::cout << "  [keep]  " << img.filename().string() << " (png, width " << width << ")\n";
            return;
        }
        cmd = "sips --resampleWidth 1600 " + q + " --out " + q + " >/dev/null";
    } else {
        cmd = (width > 1600 ? "sips --resampleWidth 1600 " : "sips ") +
              string("-s format jpeg -s formatOptions 80 ") + q + " --out " + q + " >/dev/null";
    }
    if (dryRun) {
        std::cout << "  [would compress] " << img.filename().string()
                  << " (" << before / 1024 << " KB, width " << width << ")\n";
        return;
    }
    std::system(cmd.c_str());
    auto after = fs::file_size(img);
    std::cout << "  [shrink] " << img.filename().string() << ": "
              << before / 1024 << " KB -> " << after / 1024 << " KB\n";
}

// Produce a .webp sibling next to a JPEG/PNG so posts can serve <picture>.
// Requires cwebp; silently skipped when unavailable.
static bool makeWebp(const fs::path& img) {
    static int haveCwebp = -1;
    if (haveCwebp < 0) haveCwebp = runCapture("command -v cwebp").empty() ? 0 : 1;
    if (!haveCwebp) return false;

    fs::path webp = img;
    webp.replace_extension(".webp");
    string cmd = "cwebp -quiet -q 80 -m 4 " + shellQuote(img.string()) +
                 " -o " + shellQuote(webp.string());
    return std::system(cmd.c_str()) == 0 && fs::exists(webp);
}

// ---------------------------------------------------------- image pipeline --

static std::pair<int, int> imageDims(const fs::path& img) {
    string out = runCapture("sips -g pixelWidth -g pixelHeight " + shellQuote(img.string()) +
                            " | awk '/pixel/ {print $2}'");
    int w = 0, h = 0;
    std::istringstream(out) >> w >> h;
    return {w, h};
}

struct ImageCtx {
    fs::path srcDir;    // directory of the source document
    fs::path repoRoot;
    string imgDirRel;   // e.g. images/2026-7-30-My-Post
    bool dryRun;
    std::vector<string> usedNames;
};

static bool isExternalRef(const string& p) {
    return p.rfind("http://", 0) == 0 || p.rfind("https://", 0) == 0 ||
           p.rfind("{{", 0) == 0 || p.rfind("data:", 0) == 0;
}

// copy + compress one referenced image, return rewritten <img> tag (or "" to keep original)
static string importImage(ImageCtx& ctx, const string& rawPath, const string& alt) {
    fs::path src = expandUser(rawPath);
    if (src.is_relative()) src = ctx.srcDir / src;
    std::error_code ec;
    if (!fs::exists(src, ec)) {
        std::cerr << "  [warn]  image not found, reference kept as-is: " << rawPath << "\n";
        return "";
    }
    // dedupe destination file names
    string name = src.filename().string();
    int suffix = 1;
    while (std::find(ctx.usedNames.begin(), ctx.usedNames.end(), name) != ctx.usedNames.end()) {
        name = src.stem().string() + "-" + std::to_string(++suffix) + src.extension().string();
    }
    ctx.usedNames.push_back(name);

    fs::path destDir = ctx.repoRoot / ctx.imgDirRel;
    fs::path dest = destDir / name;
    int w = 0, h = 0;
    if (ctx.dryRun) {
        std::cout << "  [would copy] " << src.string() << " -> " << ctx.imgDirRel << "/" << name << "\n";
        std::tie(w, h) = imageDims(src);  // pre-compression estimate
    } else {
        fs::create_directories(destDir);
        fs::copy_file(src, dest, fs::copy_options::overwrite_existing);
        compressImage(dest, false);
        std::tie(w, h) = imageDims(dest);
    }
    string dims = (w > 0 && h > 0)
        ? " width=\"" + std::to_string(w) + "\" height=\"" + std::to_string(h) + "\""
        : "";
    string base = "{{site.baseurl}}/" + ctx.imgDirRel + "/" + name;
    string tag = "<img src=\"" + base + "\" alt=\"" + alt +
                 "\" loading=\"lazy\" decoding=\"async\"" + dims + ">";

    // prefer WebP when a sibling exists, keeping the original as fallback
    if (!ctx.dryRun && makeWebp(dest)) {
        string webpSrc = base.substr(0, base.rfind('.')) + ".webp";
        tag = "<picture><source srcset=\"" + webpSrc + "\" type=\"image/webp\">" +
              tag + "</picture>";
    }
    return tag;
}

// rewrite ![alt](path) and <img src="path"> that point at local files
static string processImages(const string& body, ImageCtx& ctx) {
    std::regex re(R"re((!\[([^\]]*)\]\(([^)\s]+)[^)]*\))|(<img[^>]*?src="([^"]+)"[^>]*>))re");
    string out;
    auto begin = std::sregex_iterator(body.begin(), body.end(), re);
    size_t last = 0;
    for (auto it = begin; it != std::sregex_iterator(); ++it) {
        const std::smatch& m = *it;
        out += body.substr(last, m.position() - last);
        last = m.position() + m.length();

        bool isMd = m[1].matched;
        string alt  = isMd ? m[2].str() : "image";
        string path = isMd ? m[3].str() : m[5].str();
        if (isExternalRef(path)) {
            out += m.str();  // leave web / already-sited references alone
        } else {
            string tag = importImage(ctx, path, alt);
            out += tag.empty() ? m.str() : tag;
        }
    }
    out += body.substr(last);
    return out;
}

// -------------------------------------------------------- math delimiters --

// site rule: kramdown only protects $$...$$, so promote single-$ inline math.
// Skips fenced code blocks and existing $$ spans.
static string promoteInlineMath(const string& body) {
    std::istringstream in(body);
    std::ostringstream out;
    string line;
    bool inFence = false, first = true;
    while (std::getline(in, line)) {
        if (!first) out << "\n";
        first = false;
        if (trim(line).rfind("```", 0) == 0) { inFence = !inFence; out << line; continue; }
        if (inFence) { out << line; continue; }
        string res;
        for (size_t i = 0; i < line.size(); ++i) {
            if (line[i] == '$') {
                if (i + 1 < line.size() && line[i + 1] == '$') {         // existing $$...$$
                    size_t close = line.find("$$", i + 2);
                    if (close == string::npos) { res += line.substr(i); break; }
                    res += line.substr(i, close + 2 - i);
                    i = close + 1;
                } else {                                                  // single $...$
                    size_t close = line.find('$', i + 1);
                    if (close == string::npos) { res += line.substr(i); break; }
                    res += "$$" + line.substr(i + 1, close - i - 1) + "$$";
                    i = close;
                }
            } else res += line[i];
        }
        out << res;
    }
    return out.str();
}

// ------------------------------------------------------------- generation --

static string buildFrontMatter(const string& title, const string& tags, bool math,
                               const string& thumb = "", bool gallery = false) {
    std::ostringstream fm;
    fm << "---\nlayout: post\ntitle: " << title << "\ntags: [" << tags << "]\n";
    if (math) fm << "math: true\n";
    if (!thumb.empty()) fm << "thumb: " << thumb << "\n";
    if (gallery) fm << "gallery: true\n";
    fm << "---\n\n";
    return fm.str();
}

int main(int argc, char** argv) {
    Options opt = parseArgs(argc, argv);
    fs::path repoRoot = findRepoRoot();
    fs::path input = fs::absolute(expandUser(opt.input.string()));

    if (!fs::exists(input)) {
        std::cerr << "postgen: no such file or directory: " << input.string() << "\n";
        return 1;
    }

    string ext = toLower(input.extension().string());
    if (!opt.photosMode && ext != ".md" && ext != ".tex" && ext != ".pdf") {
        std::cerr << "postgen: unsupported input type " << ext << " (want .md/.tex/.pdf)\n";
        return 1;
    }

    string date = opt.date.empty() ? todayDate() : opt.date;
    string body, title = opt.title;
    bool math = opt.math;

    // ---- read + convert input, auto-extract title ----
    if (opt.photosMode) {
        if (!fs::is_directory(input)) {
            std::cerr << "postgen: --photos expects a directory\n";
            return 1;
        }
        if (title.empty()) title = input.filename().string();
    } else if (ext == ".md") {
        auto [stripped, fmTitle] = stripFrontMatter(readFile(input));
        body = stripped;
        string h1 = extractH1(body);
        body = demoteExtraH1(body);
        if (title.empty()) title = !fmTitle.empty() ? fmTitle : h1;
        if (body.find("$$") != string::npos) math = true;
    } else if (ext == ".tex") {
        if (runCapture("command -v pandoc").empty()) {
            std::cerr << "postgen: pandoc is required for .tex input (brew install pandoc)\n";
            return 1;
        }
        if (title.empty()) title = texTitle(readFile(input));
        fs::path tmp = fs::temp_directory_path() / "postgen_pandoc.md";
        // -tex_math_gfm: forbid ```math fences / $`..`$ (kramdown can't render them);
        // +tex_math_dollars: emit $...$ / $$...$$, later promoted to site-standard $$
        // --shift-heading-level-by=1: \section -> h2, matching the TOC tree root
        string cmd = "pandoc -f latex -t gfm-tex_math_gfm+tex_math_dollars --shift-heading-level-by=1 " +
                     shellQuote(input.string()) + " -o " + shellQuote(tmp.string());
        if (std::system(cmd.c_str()) != 0) {
            std::cerr << "postgen: pandoc failed to convert " << input.filename().string() << "\n";
            return 1;
        }
        body = promoteInlineMath(readFile(tmp));
        fs::remove(tmp);
        math = true;
    } else if (ext == ".pdf") {
        if (title.empty()) title = pdfTitle(input);
    }

    // ---- title: interactive fallback ----
    if (title.empty()) {
        std::cout << "Title (empty = use file name): " << std::flush;
        std::getline(std::cin, title);
        title = trim(title);
        if (title.empty()) title = input.stem().string();
    }

    string slug = slugify(title);
    string postName = date + "-" + slug;
    fs::path postPath = repoRoot / "_posts" / (postName + ".md");
    if (fs::exists(postPath) && !opt.force) {
        std::cerr << "postgen: " << postPath.string() << " already exists (use --force to overwrite)\n";
        return 1;
    }

    // ---- build body (image pipeline / pdf embed / photos) ----
    ImageCtx ctx{input.parent_path(), repoRoot, "images/" + postName, opt.dryRun, {}};

    if (opt.photosMode) {
        std::vector<fs::path> pics;
        for (auto const& e : fs::directory_iterator(input)) {
            string pe = toLower(e.path().extension().string());
            if (pe == ".jpg" || pe == ".jpeg" || pe == ".png") pics.push_back(e.path());
        }
        std::sort(pics.begin(), pics.end());
        int fid = 0;
        std::ostringstream b;
        for (auto const& p : pics) {
            string tag = importImage(ctx, p.string(), "pic" + std::to_string(fid++));
            if (!tag.empty()) b << tag << "\n\n";
        }
        body = b.str();
    } else if (ext == ".pdf") {
        string pdfRel = "assets/pdf/" + postName + ".pdf";
        fs::path pdfDest = repoRoot / pdfRel;
        if (opt.dryRun) {
            std::cout << "  [would copy] " << input.string() << " -> " << pdfRel << "\n";
        } else {
            fs::create_directories(pdfDest.parent_path());
            fs::copy_file(input, pdfDest, fs::copy_options::overwrite_existing);
        }
        body = "<embed src=\"{{site.baseurl}}/" + pdfRel +
               "\" type=\"application/pdf\" style=\"width:100%;height:80vh;border-radius:12px;\" />\n\n"
               "[Download PDF]({{site.baseurl}}/" + pdfRel + ")\n";
    } else {
        body = processImages(body, ctx);
    }

    // index-card thumbnail from the first imported image
    string thumb;
    if (!ctx.usedNames.empty()) {
        thumb = "/" + ctx.imgDirRel + "/thumb.jpg";
        fs::path first = repoRoot / ctx.imgDirRel / ctx.usedNames.front();
        fs::path thumbPath = repoRoot / ctx.imgDirRel / "thumb.jpg";
        if (opt.dryRun) {
            std::cout << "  [would create] " << thumb << "\n";
        } else {
            std::system(("sips --resampleWidth 480 -s format jpeg -s formatOptions 70 " +
                         shellQuote(first.string()) + " --out " + shellQuote(thumbPath.string()) +
                         " >/dev/null").c_str());
            makeWebp(thumbPath);  // <picture> source for the index card
        }
    }

    // photo-album posts render as a grid with the custom lightbox
    string post = buildFrontMatter(title, opt.tags, math, thumb, opt.photosMode)
                  + trim(body) + "\n";

    // paths to hand to git when --publish is set
    std::vector<string> publishPaths{"_posts/" + postName + ".md"};
    if (!ctx.usedNames.empty()) publishPaths.push_back(ctx.imgDirRel);
    if (!opt.photosMode && ext == ".pdf") publishPaths.push_back("assets/pdf/" + postName + ".pdf");

    // ---- emit ----
    if (opt.dryRun) {
        std::cout << "\n--- would write " << postPath.string() << " ---\n" << post;
        if (opt.publish) std::cout << "--- would publish (git add/commit/push) ---\n";
        return 0;
    }
    std::ofstream(postPath, std::ios::binary) << post;
    std::cout << "created " << postPath.string() << "\n";

    if (opt.publish) {
        string git = "git -C " + shellQuote(repoRoot.string());
        string add = git + " add";
        for (const auto& p : publishPaths) add += " " + shellQuote(p);
        if (std::system(add.c_str()) != 0 ||
            std::system((git + " commit -m " + shellQuote("post. " + title)).c_str()) != 0 ||
            std::system((git + " push").c_str()) != 0) {
            std::cerr << "postgen: publish failed — post was generated, fix git state and push manually\n";
            return 1;
        }
        std::cout << "published \"" << title << "\" (GitHub Pages will rebuild shortly)\n";
    }
    return 0;
}
