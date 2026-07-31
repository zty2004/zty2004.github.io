#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
namespace fs = std::filesystem;

int main() {
    const fs::path cwd = fs::current_path();  // get the current path
    int fid = -1;                             // skip this file and the markdown file

    std::ofstream outfile(cwd.filename().string() + ".md", std::ios::out);

    outfile << "---" << std::endl;
    outfile << "layout: post" << std::endl;
    outfile << "title: " << cwd.filename().string() << std::endl;
    outfile << "---" << std::endl
            << std::endl;

    for (auto const& dir_entry : fs::directory_iterator{cwd}) {  // iterate the dir
        outfile << "<img src=\"{{site.baseurl}}/"
                << cwd.parent_path().filename().string() << "/"
                << cwd.filename().string() << "/"
                << dir_entry.path().filename().string()
                << "\" alt=\"pic" << ++fid << "\" loading=\"lazy\" decoding=\"async\">" << std::endl
                << std::endl;
    }
    outfile.close();
    return 0;
}
