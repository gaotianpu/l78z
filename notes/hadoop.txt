一、了解原理
0. 应用场景
1. 大数据背景下，【输入>计算>输出】模型遇到的挑战
    计算方式：串行 -> 并行
    数据存储：单个磁盘 -> 磁盘阵列    
2. map & reduce 
    一般流程： input -> map -> shuffle -> reduce -> output
    input > splitting > mapping -> combiner > shuffling > reducing > 输出
        * 数据本地化优化
        * map 输出分区，分区函数partitioner
    http://blog.jobbole.com/84089/
    基本架构，组件 
3. hdfs
    文件操作:增删改查等等
    shell &  java api 
    namenode,datanode
    一个writer，写入到文件的末尾
    kerberos 用户认证
    fuse,挂载hdfs
    集群的网络拓扑结构
    HAR,存档工具，archive
4. 各种异常
    5.1 存储： 节点故障
    5.2 计算： 各种环节的问题以及对策
扩展
YARN的工作机制



新产生的文件，需要和原来的合并？
文件按大小拆分，处理时，通常按行？



二、伪分布式环境安装配置 
    1. 下载
        mkdir ~/Bigdata 
        cd ~/Bigdata 
        wget http://mirrors.tuna.tsinghua.edu.cn/apache/hadoop/common/hadoop-2.8.1/hadoop-2.8.1.tar.gz
        tar -zxvf hadoop-2.8.1.tar.gz

    2. ssh设置 伪分布式需要用到
        cd ~ 
        ssh-keygen -t rsa -P '' -f ~/.ssh/id_rsa
        cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
        ssh localhost 

    3. 安装JDK
        java 

    4. vim ~/.bash_profile
        export HADOOP_HOME=/Users/gaotianpu/BigData/hadoop-2.8.1
        export PATH=$PATH:$HADOOP_HOME/bin
        #export PATH=$PATH:$HADOOP_HOME/sbin #?
        
        source ~/.bash_profile  #立即生效  
        #http://elf8848.iteye.com/blog/1582137
        
    5. Hadoop伪分布式配置？
        http://blog.csdn.net/lcj369387335/article/details/45046167
        配置文件目录：etc/hadoop/
        etc/hadoop/core-site.xml
        etc/hadoop/hdfs-site.xml
        etc/hadoop/mapred-site.xml.template?

    6. 启动伪分布式
        bin/hadoop namenode -format #每次启动前必须先执行format操作？
        sbin/start-all.sh #启动进程
        jps  #确认java进程

        http://localhost:50070/ #hdfs
        http://localhost:8088/ #mapreduce 

    #系统启动时自动运行？

    #运行sample程序 

    #高级配置
    1. log目录 log文件前缀
    2. hdfs目录

   

2. 搭建java+hadoop的开发环境
    hadoop学习之二:mac下hadoop+eclipse环境搭建
    http://blog.csdn.net/Ruidu_Doer/article/details/50781144
3. 完整走一个基于pyhton的hadoop map Reduce？ 
    基于python的mapreduce开发流程
    http://blog.csdn.net/susser43/article/details/41518831   
4. 下载更多示例代码，试运行
    count,sum,max,min,去重,avg,标准差等
    排序？
    简单的跑以下sample code, 主要还是放在pig和hive上？
5. 基于hadoop的算法、机器学习等
    《Hadoop应用架构》
    《数据算法:Hadoop/Spark大数据处理技巧》

三、思考
1. 很多人并不了解mysql的各种细节，初级能力建表、sql查询等，hadoop教材应该也遵循这个规则，先初级实践，再更深层次的？
2. docker + hadoop ?
3. hadoop or spark ?
4. 知识图谱，实体+关系的存储？
5. 书籍提供了系统的框架性的学习指导，遇到某些细节书上讲解的不清楚时，再利用搜索引擎找相关文章阅读
6. 不用一行代码介绍mapreduce?
7. 网络拓扑结构？
8. hdfs 图存储
9. hbase 知识图谱存储？

四、随笔
分区数=作业的Reducer数？
按大小分片，按行处理，某行跨文件块
数据采样？
join 关联
先上传数据，再打包上传mr 程序？
RAID是英文Redundant Array of Independent Disks的缩写,翻译成中文意思是“独立磁盘冗余阵列”,有时也简称磁盘阵列(Disk Array)
nfs?
hdfs mapreduce独立安装？ namenode 运行hdfs控制脚本,jobtracker运行map reduce脚本
日志目录独立
ssh的系统知识？

Flume, 分布式日志收集系统
Sqoop,
Avro,序列化？

基于docker部署hadoop？


五、问题
1.Q:WARN util.NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
A: 可以不解决
解决办法：download the source code of Hadoop and recompile libhadoop.so.1.0.0 on 64bit system, then replace the 32bit one.

2.Incorrect configuration: namenode address dfs.namenode.servicerpc-address or dfs.namenode.rpc-address is not configured.
缺少配置文件

3. localhost: /Users/gaotianpu/.bashrc: line 1: brew: command not found
which brew
 ~ /usr/local/bin/brew
 export PATH=/usr/local/bin:$PATH

job - task 

job调度
1. fifo
2. fair scheduler , task 槽的数量，支持【抢占】


shuffle

环形内存缓冲池

#etc/hadoop/hdfs-site.xml
<property>  
        <name>dfs.replication</name>  
        <value>1</value>  
    </property>  
    <property>  
        <name>dfs.namenode.name.dir</name>  
        <value>/Users/gaotianpu/Bigdata/dfs/name</value>  
    </property>  
    <property>  
        <name>dfs.datannode.data.dir</name>  
        <value>/Users/gaotianpu/Bigdata/dfs/data</value>  
    </property> 


#etc/hadoop/mapred-site.xml
 <property>  
        <name>fs.default.name</name>  
        <value>hdfs://localhost:9000</value>  
    </property>  
    <property>  
        <name>mapred.job.tracker</name>  
        <value>hdfs://localhost:9001</value>  
    </property>  
    <property>  
        <name>dfs.replication</name>  
        <value>1</value>  
    </property>   