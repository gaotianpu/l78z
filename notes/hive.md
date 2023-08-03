hive
1. 下载安装
    http://hive.apache.org
    https://cwiki.apache.org/confluence/display/Hive/GettingStarted

2. 配置
    http://nekomiao.me/2016/10/11/mac-install-hive/

    #JDBC驱动
    cp mysql-connector-java-5.1.43-bin.jar $HIVE_HOME/lib/mysql-connector-java-5.1.43-bin.jar
    cp $HIVE_HOME/lib/jline-2.12.jar  $HADOOP_HOME/share/hadoop/yarn/lib/jline-2.12.jar


    cp ./hive-default.xml.template ./hive-default.xml
    touch hive-site.xml

    cp hive-log4j.properties.template hive-log4j.properties
    cp hive-env.sh.template hive-env.sh

    https://codetosurvive1.github.io/posts/install-hive-basic.html

     

3. 初始化
    cd $HIVE_HOME/scripts/metastore/upgrade/mysql/

    < Login into MySQL >

    mysql> drop database IF EXISTS hive;
    mysql> create database hive;
    mysql> use hive;
    mysql> source hive-schema-2.1.1.mysql.sql;

    hive --service metastore
    https://stackoverflow.com/questions/42209875/hive-2-1-1-metaexceptionmessageversion-information-not-found-in-metastore

http://f.dataguru.cn/thread-288208-1-1.html

4. 常见错误
    Q：启动hive时报错
    SLF4J: Class path contains multiple SLF4J bindings.
    SLF4J: Found binding in [jar:file:/Users/bxd/Bigdata/apache-hive-2.3.0-bin/lib/log4j-slf4j-impl-2.6.2.jar!/org/slf4j/impl/StaticLoggerBinder.class]
    SLF4J: Found binding in [jar:file:/Users/bxd/Bigdata/hadoop-2.8.1/share/hadoop/common/lib/slf4j-log4j12-1.7.10.jar!/org/slf4j/impl/StaticLoggerBinder.class]
    SLF4J: See http://www.slf4j.org/codes.html#multiple_bindings for an explanation.
    SLF4J: Actual binding is of type [org.apache.logging.slf4j.Log4jLoggerFactory]
    A： https://stackoverflow.com/questions/27050820/running-hive-0-12-with-error-of-slf4j
    rm lib/hive-jdbc-2.0.0-standalone.jar
    rm lib/log4j-slf4j-impl-2.4.1.jar

    Logging initialized using configuration in jar:file:/Users/bxd/Bigdata/apache-hive-2.3.0-bin/lib/hive-common-2.3.0.jar!/hive-log4j2.properties Async: true
    Hive-on-MR is deprecated in Hive 2 and may not be available in the future versions. Consider using a different execution engine (i.e. spark, tez) or using Hive 1.X releases.

    Q:从hive中退出
    Wed Aug 02 13:05:33 CST 2017 WARN: Establishing SSL connection without server's identity verification is not recommended. According to MySQL 5.5.45+, 5.6.26+ and 5.7.6+ requirements SSL connection must be established by default if explicit option isn't set. For compliance with existing applications not using SSL the verifyServerCertificate property is set to 'false'. You need either to explicitly disable SSL by setting useSSL=false, or set useSSL=true and provide truststore for server certificate verification.
    Q: jdbc:mysql://localhost:3306/hive?createDatabaseIfNotExist=true&amp;useSSL=false

    3. 初始化
    schematool -dbType mysql -initSchema

    org.apache.hadoop.hive.metastore.HiveMetaException: Failed to get schema version.
    Underlying cause: java.sql.SQLException : Failed to start database 'metastore_db' with class loader sun.misc.Launcher$AppClassLoader@282ba1e, 
    see the next exception for details.
    SQL Error code: 40000
    Use --verbose for detailed stacktrace.

    Initialization script hive-schema-2.3.0.mysql.sql
    Exception in thread "main" java.lang.NoClassDefFoundError: org/apache/hive/jdbc/logs/InPlaceUpdateStream
        at org.apache.hive.beeline.BeeLine.<init>(BeeLine.java:136)
        at org.apache.hive.beeline.BeeLine.<init>(BeeLine.java:530)
        at org.apache.hive.beeline.HiveSchemaTool.runBeeLine(HiveSchemaTool.java:967)
        at org.apache.hive.beeline.HiveSchemaTool.runBeeLine(HiveSchemaTool.java:959)
        at org.apache.hive.beeline.HiveSchemaTool.doInit(HiveSchemaTool.java:586)
        at org.apache.hive.beeline.HiveSchemaTool.doInit(HiveSchemaTool.java:563)
        at org.apache.hive.beeline.HiveSchemaTool.main(HiveSchemaTool.java:1145)
        at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
        at sun.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:62)
        at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
        at java.lang.reflect.Method.invoke(Method.java:498)
        at org.apache.hadoop.util.RunJar.run(RunJar.java:234)
        at org.apache.hadoop.util.RunJar.main(RunJar.java:148)
    Caused by: java.lang.ClassNotFoundException: org.apache.hive.jdbc.logs.InPlaceUpdateStream
        at java.net.URLClassLoader.findClass(URLClassLoader.java:381)
        at java.lang.ClassLoader.loadClass(ClassLoader.java:424)
        at sun.misc.Launcher$AppClassLoader.loadClass(Launcher.java:335)
        at java.lang.ClassLoader.loadClass(ClassLoader.java:357)
        ... 13 more

    hive> show databases;
    FAILED: SemanticException org.apache.hadoop.hive.ql.metadata.HiveException: java.lang.RuntimeException: Unable to instantiate org.apache.hadoop.hive.ql.metadata.SessionHiveMetaStoreClient
    A：mysql -uroot -p 
    drop database hive



    http://www.jianshu.com/p/d9cb284f842d
    http://www.qinbin.me/mac-osx%E5%AE%89%E8%A3%85apache-hive/



    http://blog.csdn.net/freedomboy319/article/details/44828337
    hive --service metastore &
    Opening raw store with implementation class:org.apache.hadoop.hive.metastore.ObjectStore
    MetaException(message:Version information not found in metastore. )

    https://stackoverflow.com/questions/35449274/java-lang-runtimeexception-unable-to-instantiate-org-apache-hadoop-hive-ql-meta


HiveQL -> map/reduce作业
糟糕的查询语句会导致执行效率降低

客户端输入sql - map/reduce作业->集群 - 返回结果

hadoop fs -mkdir /tmp
hadoop fs -chmod g+w /tmp

1. 定义表schema
2. 导入
3. 执行HiveQL查询

MetaStore

Exception in thread "main" java.lang.RuntimeException: org.apache.hadoop.hive.ql.metadata.HiveException: java.lang.RuntimeException: Unable to instantiate org.apache.hadoop.hive.ql.metadata.SessionHiveMetaStoreClient

分区
桶，还不好理解
列存储，不好理解

多表插入，一次读取，多处写入

metastore?
thrift 

元数据的存储？


