 如何让512内存的上网本还能继续使用？
 https://www.archlinux.org/
 https://wiki.archlinux.org/index.php/Installation_guide_(%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87)
 
 http://blog.sina.com.cn/s/blog_87113ac20102w8w2.html


 http://www.cnblogs.com/vachester/p/5635819.html
 


linux-shell

编写可靠shell脚本的八个建议
http://ju.outofmemory.cn/entry/280805
http://151wqooo.blog.51cto.com/2610898/1174066 

Google Shell Style Guide
https://google.github.io/styleguide/shell.xml


linux常用命令
1. pwd, 获得当前目录名


2.
scp


3.
ls -l | wc -l

update-alternatives --config editor 

设置时区 gmt+8

设置系统时间
#date -s 08/08/2008
或
#date -s 20080808
将系统时间设定成下午8点8分0秒的命令如下。
#date -s 20:08:00

set -e 和 set -x

~~~
# if 
if [!-d "./log"] then
fi

# case 
case x in 
a)
    fun()
    ;;
*)

# for
for x in $1;
do
    echo $x
done

~~~

文件格式转换
iconv -f gbk -t utf-8 empty.txt.2017-06-13 -o empty.txt

awk
cat nbr.txt | awk -F '\t' '{if($3==0) print $0}' | wc -l

scp -r  local_dir  root@ip:remote_dir
rsync -av local_dir --exclude=log root@ip:remote_dir

find  ./* -type f -mtime +3 -exec rm {} \;

date "+%Y-%m-%d %H:%M:%S"

wget -P 

mkdir -p

exit 1 

esac ?

du -sh filename #查看文件大小

cp -avx /home/* /mnt/newhome  #复制整个目录和文件

#rsync ?
远程机器没开samba，直接在机器上使用vim非强项。
开发机上设置rsync,文件有更新实时同步？


date -d 1504155900  #shell中时间戳转date

date -r 1504673100

echo 9230225434485993258 | wc -L #统计字符长度

shell 条件判断
http://blog.csdn.net/yf210yf/article/details/9207147

linux常用命令
1. pwd, 获得当前目录名


2.
scp


3.
ls -l | wc -l

update-alternatives --config editor 

设置时区 gmt+8

设置系统时间
#date -s 08/08/2008
或
#date -s 20080808
将系统时间设定成下午8点8分0秒的命令如下。
#date -s 20:08:00

1. 启动默认进入命令行模式
sudo vi /etc/default/grub

GRUB_CMDLINE_LINUX_DEFAULT="quiet splash text"  #add text

sudo update-grub

2. 在命令行模式下进入x-windows ？？？？
startx 有问题，仅显示一个壁纸，没有状态条等
ALT+CTRL+F7

3. 从x-windows进入命令行模式
ALT+CTRL+F1

4.ubuntu命令行下中文乱码
方案1. 默认显示英文
sudo vim /etc/default/locale
LANG="en_US.UTF-8"   #h_CN.UTF-8
LANGUAGE="en_US:en"  #zh_CN:zh

方案2. 支持中文
sudo apt-get install zhcon
zhcon --utf8 --drv=vga
别名
sudo vim ~/.bashrc
alias zhcon='zhcon --utf8 --drv=vga'



4. 关机
shutdown -h now  #关机 不带-h就不会关电源
重启 reboot
待机 pm-suspend
休眠 pm-hibernate

5.配置：设置固定ip？
sudo vi /etc/network/interfaces
同一个局域网的，不用记ip，而是知道机器名字即可用ssh登录的


6.配置：允许远程登录
sudo apt-get install openssh-server
sudo vi /etc/ssh/sshd_config
PermitRootLogin yes
sudo service ssh restart


7.开发环境-comm
sudo apt-get install vim  #
sudo apt-get install python-dev
sudo apt-get install python-setuptools
sudo apt-get install mysql
sudo apt-get install sqlite
sudo apt-get install redis
sudo apt-get install php5-cli
sudo apt-get install nginx

8. 开发环境-python
sudo apt-get install python-mysqldb
easy_install web.py
apt-get install python-numpy
apt-get install python-scipy

ssh免密码登录
http://chenlb.iteye.com/blog/211809

sudo apt-get install ntpdate
sudo ntpdate cn.pool.ntp.org

tzselect无效
sudo raspi-config
chongqing

http://blog.csdn.net/jdh99/article/details/22096479

scp  /Users/gaotianpu/github/forecast/v2/daily_run.py  pi@192.168.1.100:/home/pi/workspace/stocks


1. 启动默认进入命令行模式
sudo vi /etc/default/grub

GRUB_CMDLINE_LINUX_DEFAULT="quiet splash text"  #add text

sudo update-grub

2. 在命令行模式下进入x-windows ？？？？
startx 有问题，仅显示一个壁纸，没有状态条等
ALT+CTRL+F7

3. 从x-windows进入命令行模式
ALT+CTRL+F1

4.ubuntu命令行下中文乱码
方案1. 默认显示英文
sudo vim /etc/default/locale
LANG="en_US.UTF-8"   #h_CN.UTF-8
LANGUAGE="en_US:en"  #zh_CN:zh

方案2. 支持中文
sudo apt-get install zhcon
zhcon --utf8 --drv=vga
别名
sudo vim ~/.bashrc
alias zhcon='zhcon --utf8 --drv=vga'



4. 关机
shutdown -h now  #关机 不带-h就不会关电源
重启 reboot
待机 pm-suspend
休眠 pm-hibernate

5.配置：设置固定ip？
sudo vi /etc/network/interfaces
同一个局域网的，不用记ip，而是知道机器名字即可用ssh登录的


6.配置：允许远程登录
sudo apt-get install openssh-server
sudo vi /etc/ssh/sshd_config
PermitRootLogin yes
sudo service ssh restart


7.开发环境-comm
sudo apt-get install vim  #
sudo apt-get install python-dev
sudo apt-get install python-setuptools
sudo apt-get install mysql
sudo apt-get install sqlite
sudo apt-get install redis
sudo apt-get install php5-cli
sudo apt-get install nginx

8. 开发环境-python
sudo apt-get install python-mysqldb
easy_install web.py
apt-get install python-numpy
apt-get install python-scipy

ssh免密码登录
http://chenlb.iteye.com/blog/211809

sudo apt-get install ntpdate
sudo ntpdate cn.pool.ntp.org

tzselect无效
sudo raspi-config
chongqing

http://blog.csdn.net/jdh99/article/details/22096479

scp  /Users/gaotianpu/github/forecast/v2/daily_run.py  pi@192.168.1.100:/home/pi/workspace/stocks