一、下载安装
1. ubuntu, apt-get
dpkg -S /usr/include/boost/version.hpp  #查找已安装的版本
sudo find /usr -name boost  #查找安装位置
sudo apt-get install libboost-all-dev
aptitude search boost
apt-cache search boost

// remove the dirs under <prefix>/lib/libboost* and <prefix>/include/boost* first
sudo apt purge -y libboost-all-dev libboost*
sudo apt autoremove

sudo ldconfig #?

2. 源代编译安装
https://www.boost.org/users/download/

mac clang 
https://solarianprogrammer.com/2018/08/07/compiling-boost-gcc-clang-macos/


cd /mnt/d/gitee1/
wget https://boostorg.jfrog.io/artifactory/main/release/1.80.0/source/boost_1_80_0.tar.gz
tar -xzvf boost_1_80_0.tar.gz
cd boost_1_80_0
./bootstrap.sh --prefix=/usr/local/boost_1_80_0
sudo ./b2 cxxflags=-std=c++17 install

export DYLD_LIBRARY_PATH=/usr/local/boost_1_80_0/lib:$DYLD_LIBRARY_PATH


二、使用
export DYLD_LIBRARY_PATH=/usr/local/lib/:$DYLD_LIBRARY_PATH
g++ -std=c++17 -o main main.cpp -L /usr/local/lib/ -lboost_system  -lboost_filesystem
# -I /usr/include/boost/ 

cmake .  && make && ./main.o
 

./main
./main: error while loading shared libraries: libboost_filesystem.so.1.80.0: cannot open shared object file: No such file or directory

g++ -std=c++17 -o main main.cpp -lboost_system  -lboost_filesystem
/usr/bin/ld: /tmp/ccMLNSsB.o: in function `boost::filesystem::directory_iterator::directory_iterator(boost::filesystem::path const&, boost::filesystem::directory_options)':
main.cpp:(.text._ZN5boost10filesystem18directory_iteratorC2ERKNS0_4pathENS0_17directory_optionsE[_ZN5boost10filesystem18directory_iteratorC5ERKNS0_4pathENS0_17directory_optionsE]+0x3e): undefined reference to `boost::filesystem::detail::directory_iterator_construct(boost::filesystem::directory_iterator&, boost::filesystem::path const&, unsigned int, boost::filesystem::detail::directory_iterator_params*, boost::system::error_code*)'
/usr/bin/ld: /tmp/ccMLNSsB.o: in function `void boost::sp_adl_block::intrusive_ptr_release<boost::filesystem::detail::dir_itr_imp, boost::sp_adl_block::thread_safe_counter>(boost::sp_adl_block::intrusive_ref_counter<boost::filesystem::detail::dir_itr_imp, boost::sp_adl_block::thread_safe_counter> const*)':
main.cpp:(.text._ZN5boost12sp_adl_block21intrusive_ptr_releaseINS_10filesystem6detail11dir_itr_impENS0_19thread_safe_counterEEEvPKNS0_21intrusive_ref_counterIT_T0_EE[_ZN5boost12sp_adl_block21intrusive_ptr_releaseINS_10filesystem6detail11dir_itr_impENS0_19thread_safe_counterEEEvPKNS0_21intrusive_ref_counterIT_T0_EE]+0x33): undefined reference to `boost::filesystem::detail::dir_itr_imp::~dir_itr_imp()'
/usr/bin/ld: main.cpp:(.text._ZN5boost12sp_adl_block21intrusive_ptr_releaseINS_10filesystem6detail11dir_itr_impENS0_19thread_safe_counterEEEvPKNS0_21intrusive_ref_counterIT_T0_EE[_ZN5boost12sp_adl_block21intrusive_ptr_releaseINS_10filesystem6detail11dir_itr_impENS0_19thread_safe_counterEEEvPKNS0_21intrusive_ref_counterIT_T0_EE]+0x3b): undefined reference to `boost::filesystem::detail::dir_itr_imp::operator delete(void*)'
collect2: error: ld returned 1 exit status



