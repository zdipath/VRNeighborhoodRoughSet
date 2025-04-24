%function iris_data=irisfunctionpredp()
function [iris_data,nullm,discreta,cla]=irisfunctionpredp()
%只适用于没有缺省且数值型条件属性
% 整理iris数据集
%m为元素的个数
m=150;
cla=3;
%n为条件属性加决策属性的个数
n=5;
%n1，n2，，nr表示r个决策类分别对应的对象个数
% f=fopen('iris.txt');
f=fopen('iris.data');% 打开文件
data=textscan(f,'%f,%f,%f,%f,%s'); % 读取数据
nullm=zeros(m,n);
discreta=[0,0,0,0];
D=[];% D中存放属性值
for i=1:length(data)-1
    D=[D data{1,i}];
end
fclose(f);


lable=data{1,length(data)};
n1=0;n2=0;n3=0;
% 找到每类数据的索引
for j=1:length(lable)
   if strcmp(lable{j,1},'Iris-setosa')
       n1=n1+1;
       index_1(n1)=j;% 记录下属于“Iris-setosa”类的索引
       
   elseif strcmp(lable{j,1},'Iris-versicolor')
       n2=n2+1;
       index_2(n2)=j;
       
   elseif strcmp(lable{j,1},'Iris-virginica')
       n3=n3+1;
       index_3(n3)=j;
       
   end
end

% 按照索引取出每类数据，重新组合
class_1=D(index_1,:);
class_2=D(index_2,:);
class_3=D(index_3,:);
Attributes=[class_1;class_2;class_3];

I=[1*ones(n1,1);2*ones(n2,1);3*ones(n3,1)];
Iris=[I Attributes];% 为各类添加数字标记






f=fopen('iris1.txt','w');
[m,n]=size(Iris);
for i=1:m
    for j=1:n
        if j==n
            fprintf(f,'%g \n',Iris(i,j));
        else
             fprintf(f,'%g,',Iris(i,j));
        end
    end
end

fclose(f);
%得到的结果
iris_data=load('iris1.txt');
lable_iris=iris_data(:,1);
attributes_iris=iris_data(:,2:end);
% save iris.txt -ascii Iris 
% dlmwrite('iris.txt',Iris);

















end