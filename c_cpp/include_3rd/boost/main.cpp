#include <iostream>
#include <boost/filesystem.hpp>

int main()
{
    // Get the current directory
    auto path = boost::filesystem::current_path();
    std::cout << path << "\n";

    // Print the content of the current directory
    for (auto &entry : boost::filesystem::directory_iterator(path))
    {
        std::cout << entry << std::endl;
    }

    return 0;
}