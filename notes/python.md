一、python环境
anconda.txt
https://www.jianshu.com/p/d2e15200ee9b

1.下载安转
vim ~/.bash_profile

# added by Anaconda3 5.2.0 installer
export PATH="/anaconda3/bin:$PATH"

source ~/.bash_profile

2. 列出所有的环境
conda info --envs
> base

3. 激活环境
source activate base  #py3 ?
source activate py2

4. 安装包 
conda install pytorch torchvision -c pytorch

conda search opencv
conda install opencv

conda update opencv

5. 启动notebook
jupyter notebook


6.导入/导出
conda env export > environment.yaml
conda env create -f environment.yaml 