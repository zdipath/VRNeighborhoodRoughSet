%function [ACC,stade,dependent,num_attribute,aver_num_attribute]= threeVRNRknn(data,nullm,discreta,cla,h,hh,theta,ttt)
%b为阈值大小
%clc%变函数后删除
%clear%变函数后删除


 [data,nullm,discreta,cla] =  irisfunctionpredp();%修改调用函数
 h=0.16;%初始阈值
 hh=0.01;%步长
 theta=0.01;
 ttt=10;

%ttt=5;
[data_r, data_c] = size(data);
ACC=zeros(1,ttt);
m=data_r;%m为元素的个数
num=0.9*m;
n=data_c;%n为条件属性加决策属性的个数

f=0;%控制最后的ACC
dependent=zeros(10,ttt);
stade=zeros(1,ttt);
num_attribute=zeros(10,ttt);%属性数量统计
indices = crossvalind('Kfold', data_r, 10);%具体的数字表示分到第几组
for b=h:hh:h+hh*(ttt-1)%最多可以19个
f=f+1



accracy=zeros(1,10);%计算10次的精确度值
ACCRACY=0;%计算10次的平均精确度值
%将数据样本随机分割为10部分

parfor o = 1 : 10%为10
    % 获取第i份测试数据的索引逻辑值
    test = (indices == o);
    % 取反，获取第i份训练数据的索引逻辑值
    train = ~test;
    %1份测试，9份训练
    test_data = data(test, 2 : data_c);
    test_label = data(test, 1);
    train_nullm=nullm(train,1:data_c-1);
    train_data = data(train, 2 : data_c );
    train_label = data(train, 1);
    % 使用数据的代码
    tsdata=[test_label test_data];%将测试集合并成信息表的形式
    trdata=[train_label train_data];
    
    
    %进行knn验算
    %首先建立粗糙集模型                                  


min_attribute=zeros(1,n-1);%存放每个属性的最大最小值
max_attribute=zeros(1,n-1);
posdata=zeros(num,n);%正域构成的数据集
%dep=zeros(n-1,num);%1为边界域，0为正域




for j=1:n-1
    min_attribute(j)=trdata(1,j+1);%初始值为第一个对象的值
    max_attribute(j)=trdata(1,j+1);
end
for i=1:num%进行最大最小值的查找
    for j=1:n-1
        if discreta(j)==0
            if trdata(i,j+1)<min_attribute(1,j) && train_nullm(i,j)~=1
                min_attribute(1,j)=trdata(i,j+1);
            elseif  trdata(i,j+1)>max_attribute(1,j)&& train_nullm(i,j)~=1
                max_attribute(1,j)=trdata(i,j+1);
            else
               
            end     
        end   
    end
end
%检验最大值与最小值是否相同
for j=1:n-1
    if discreta(j)==0
    if max_attribute(1,j)==min_attribute(1,j)
        max_attribute(1,j)=max_attribute(1,j)+1;
    end
    end
end




pos=zeros(1,num);%记录每次除去上一层正域后的元素 1表示已经被去除掉了
cp=zeros(1,n-1);%候选池  1表示已经被选择，0表示没有被选择
cporder=zeros(1,n-1);%第一个元素表示第一个被放入候选属性池的属性
s=0;%s用来计算是正域的个数
%计算i与k之间的距离并放入矩阵M中
temp1dep=ones(1,num);%临时正域,每层最优属性对应的正域  1表示非正域
for j=1:n-1 %折叠法  每层属性都需要计算一遍 
    % 如果 j=2 代表只计算前两个
    max_s=0;%每层候选中最大的正域数。
    max_cord=0;%每层候选中最大的正域数对应的是哪个属性
    %ss=zeros(1,n-1);%每层候选属性的正域个数比较
    for jjj=1:n-1  %每层折叠都需要选择一个最优的属性
        if cp(1,jjj)==0
            cporder(1,j)=jjj;
    M=zeros(num,num);%距离矩阵
    N=zeros(num,num);%邻域矩阵
    VN=zeros(num,num);%可变邻域矩阵
    label_s=zeros(num,cla);%每个对象邻域内标签的情况
    c=zeros(1,num);%表示邻域半径的大小
    neighcount=zeros(1,num);%邻域内元素的个数
    classsum=zeros(1,num);%公式3-2的求和
    classcount=zeros(1,num);%每一个对象邻域的类的种类
    ss=0;%每层候选属性的正域个数比较
    tempdep=zeros(1,num);%临时正域，这个是每一层都要选择时，才会用到的
    
    for i=1:num
        if pos(1,i)==0%没有被上一层选中
        for k=1:num
        for jj=1:j%距离求和，第几层就有几个属性
            d=0;
            if discreta(cporder(1,jj))==0%这里的cporder(1,jj)可以用jjj替代
                if  train_nullm(i,cporder(1,jj))==1 || train_nullm(k,cporder(1,jj))==1
                    d=1;     
                else
                    d=abs(trdata(i,cporder(1,jj)+1)-trdata(k,cporder(1,jj)+1))/(max_attribute(1,cporder(1,jj))-min_attribute(1,cporder(1,jj)));
                end
            elseif trdata(i,cporder(1,jj)+1)==trdata(k,cporder(1,jj)+1) && train_nullm(k,cporder(1,jj))~=1
                d=0;      
            elseif trdata(i,cporder(1,jj)+1)~=trdata(k,cporder(1,jj)+1) || train_nullm(i,cporder(1,jj))==1 || train_nullm(k,cporder(1,jj))==1
                d=1;  
            end
             M(i,k)=M(i,k)+(d)^2;
        end
        M(i,k)=sqrt (M(i,k));
        
        end
        end
    end
    %自身与自身距离改为0 （因为缺省值等于1的原因）
    for i=1:num
        M(i,i)=0;
    end
 for i=1:num%计算邻域矩阵N(x) 1为是x的邻域
     if pos(1,i)==0%没有被上一层选中
         for k=1:num
              if M(i,k)<=b
                  N(i,k)=1;
              end    
          end
     end
end   
for i=1:num%判断其邻域内的标签情况
    if pos(1,i)==0%没有被上一层选中
    for k=1:num
        if N(i,k)==1
            label_s(i,trdata(k,1))=label_s(i,trdata(k,1))+1;
            neighcount(1,i)=neighcount(1,i)+1;
        end
    end
    end
end    
 %需要添加一个3-2求和的公式
for i=1:num
    if pos(1,i)==0%没有被上一层选中
     for k=1:cla-1
        if label_s(i,k)>0
            for jj=k+1:cla
                if label_s(i,jj)>0
                    classsum(1,i)=classsum(1,i)+abs(label_s(i,k)-label_s(i,jj));
                end
            end
        end
     end
    end
end
for i=1:num
    if pos(1,i)==0%没有被上一层选中
    for k=1:cla
        if label_s(i,k)>0
            classcount(1,i)=classcount(1,i)+1;%每一个对象邻域的类的种类
        end
    end
    end
end
for i=1:num
    if pos(1,i)==0%没有被上一层选中
    if label_s(i,trdata(i,1))==neighcount(1,i)%相等则不需要变邻域
        c(i)=b;
    else
        c(i)=b*exp(-0.5*classsum(1,i)/nchoosek(classcount(1,i),2)/(neighcount(1,i)));%公式3-2   
    end  
    end
end   
for i=1:num%计算邻域矩阵VN(x) 1为是x的可变邻域
    if pos(1,i)==0%没有被上一层选中
    for k=1:num
        if M(i,k)<=c(i)
            VN(i,k)=1;
        end    
    end
    end
end
for i=1:num%计算是否为正域
    if pos(1,i)==0%没有被上一层选中
    for k=1:num
        if VN(i,k)==1&&trdata(i,1)~=trdata(k,1)
            tempdep(1,i)=1;%1为非正域
        end    
    end
    end
end
 %tempdep      测试用
for i=1:num
    if pos(1,i)==0%没有被上一层选中
    if tempdep(1,i)==0
        
        ss=ss+1;
    end 
    end
end
if ss>max_s%判断是否是最优的属性
    max_s=ss;
    max_cord=jjj;
    for i=1:num
        if pos(1,i)==0%没有被上一层选中
            temp1dep(1,i)=1;%归1化 为1 说明为非正域
            temp1dep(1,i)=tempdep(1,i);%找到正域并等于0
        end
    end
end

        end%对应的if cp（1，i）==0
    end
    %o
    %max_s
    if max_cord==0 || max_s<theta*num%这一层没有正域，在实验中将会停止。
        cporder(1,j)=0;
        break %跳出折叠法  
    else
         cporder(1,j)=max_cord;%将max_cord计算为最优的属性
         cp(1,max_cord)=1;
    end
    
for i=1:num%这一层最优的一个属性。
    if pos(1,i)==0%没有被上一层选中
    if temp1dep(1,i)==0
        
        s=s+1;
        pos(1,i)=1;%此时i不需要进行下一次运算了.
        posdata(s,:)=trdata(i,:);
    end 
    end
end
    
    
end
%cporder

for i=1:n-1
    if cp(1,i)==1
        
        num_attribute(o,f)=num_attribute(o,f)+1;
    end
end
dependent(o,f)=s/num;      %依赖度










if s<=3
    s

    ACC
    
end


    
    
    %计算knn得出分类精确
    K=3;%表示5NN
    TM=zeros(m-num,s);%训练集与测试集之间的距离
    knnlabel=zeros(m-num,K);%存放最近的三个标签
    re=zeros(1,m-num);%存放knn后的结果
    for i=1:m-num
        for k=1:s
            for j=1:n-1%j为属性
                if cp(1,j)==1
                dd=0;
                if  train_nullm(i,j)==1 || train_nullm(k,j)==1
                    dd=1;
                elseif discreta(j)==0 
                    dd=abs(tsdata(i,j+1)-posdata(k,j+1))/(max_attribute(1,j)-min_attribute(1,j));
                elseif tsdata(i,j+1)==posdata(k,j+1)
                    dd=0;      
                elseif tsdata(i,j+1)~=posdata(k,j+1)
                    dd=1; 
                end
            TM(i,k)=TM(i,k)+(dd)^2;
                end
            end
            TM(i,k)=sqrt (TM(i,k)); 
        end
        
    end
    %接下来寻找最近的k个点
    sortTM=TM';%列表示测试元素，行表示训练元素
    [Y,I]=sort(sortTM);
    for i=1:m-num
        for k=1:K
            knnlabel(i,k)=posdata(I(k,i),1);
        end
    end
    for i=1:m-num
        re(1,i)=mode(knnlabel(i,1:K));
        if re(1,i)==tsdata(i,1)
            accracy(1,o)=accracy(1,o)+1;%+1表示对了一个
        end
    end
    
end
 accracy_std=zeros(1,10);
for o=1:10
ACCRACY=accracy(1,o)+ACCRACY;
accracy_std(1,o)=100*accracy(1,o)/((m-num));
end



aver_num_attribute=mean(num_attribute,1);
 stade(1,f)=std(accracy_std,0);
ACC(1,f)=10*ACCRACY/((m-num));
end





%end