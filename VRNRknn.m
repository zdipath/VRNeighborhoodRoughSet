%function [ACC,stade,dependent,num_attribute,aver_num_attribute]= threeVRNRknn(data,nullm,discreta,cla,h,hh,theta,ttt)
%bΪ��ֵ��С
%clc%�亯����ɾ��
%clear%�亯����ɾ��


 [data,nullm,discreta,cla] =  irisfunctionpredp();%�޸ĵ��ú���
 h=0.16;%��ʼ��ֵ
 hh=0.01;%����
 theta=0.01;
 ttt=10;

%ttt=5;
[data_r, data_c] = size(data);
ACC=zeros(1,ttt);
m=data_r;%mΪԪ�صĸ���
num=0.9*m;
n=data_c;%nΪ�������ԼӾ������Եĸ���

f=0;%��������ACC
dependent=zeros(10,ttt);
stade=zeros(1,ttt);
num_attribute=zeros(10,ttt);%��������ͳ��
indices = crossvalind('Kfold', data_r, 10);%��������ֱ�ʾ�ֵ��ڼ���
for b=h:hh:h+hh*(ttt-1)%������19��
f=f+1



accracy=zeros(1,10);%����10�εľ�ȷ��ֵ
ACCRACY=0;%����10�ε�ƽ����ȷ��ֵ
%��������������ָ�Ϊ10����

parfor o = 1 : 10%Ϊ10
    % ��ȡ��i�ݲ������ݵ������߼�ֵ
    test = (indices == o);
    % ȡ������ȡ��i��ѵ�����ݵ������߼�ֵ
    train = ~test;
    %1�ݲ��ԣ�9��ѵ��
    test_data = data(test, 2 : data_c);
    test_label = data(test, 1);
    train_nullm=nullm(train,1:data_c-1);
    train_data = data(train, 2 : data_c );
    train_label = data(train, 1);
    % ʹ�����ݵĴ���
    tsdata=[test_label test_data];%�����Լ��ϲ�����Ϣ�����ʽ
    trdata=[train_label train_data];
    
    
    %����knn����
    %���Ƚ����ֲڼ�ģ��                                  


min_attribute=zeros(1,n-1);%���ÿ�����Ե������Сֵ
max_attribute=zeros(1,n-1);
posdata=zeros(num,n);%���򹹳ɵ����ݼ�
%dep=zeros(n-1,num);%1Ϊ�߽���0Ϊ����




for j=1:n-1
    min_attribute(j)=trdata(1,j+1);%��ʼֵΪ��һ�������ֵ
    max_attribute(j)=trdata(1,j+1);
end
for i=1:num%���������Сֵ�Ĳ���
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
%�������ֵ����Сֵ�Ƿ���ͬ
for j=1:n-1
    if discreta(j)==0
    if max_attribute(1,j)==min_attribute(1,j)
        max_attribute(1,j)=max_attribute(1,j)+1;
    end
    end
end




pos=zeros(1,num);%��¼ÿ�γ�ȥ��һ��������Ԫ�� 1��ʾ�Ѿ���ȥ������
cp=zeros(1,n-1);%��ѡ��  1��ʾ�Ѿ���ѡ��0��ʾû�б�ѡ��
cporder=zeros(1,n-1);%��һ��Ԫ�ر�ʾ��һ���������ѡ���Գص�����
s=0;%s��������������ĸ���
%����i��k֮��ľ��벢�������M��
temp1dep=ones(1,num);%��ʱ����,ÿ���������Զ�Ӧ������  1��ʾ������
for j=1:n-1 %�۵���  ÿ�����Զ���Ҫ����һ�� 
    % ��� j=2 ����ֻ����ǰ����
    max_s=0;%ÿ���ѡ��������������
    max_cord=0;%ÿ���ѡ��������������Ӧ�����ĸ�����
    %ss=zeros(1,n-1);%ÿ���ѡ���Ե���������Ƚ�
    for jjj=1:n-1  %ÿ���۵�����Ҫѡ��һ�����ŵ�����
        if cp(1,jjj)==0
            cporder(1,j)=jjj;
    M=zeros(num,num);%�������
    N=zeros(num,num);%�������
    VN=zeros(num,num);%�ɱ��������
    label_s=zeros(num,cla);%ÿ�����������ڱ�ǩ�����
    c=zeros(1,num);%��ʾ����뾶�Ĵ�С
    neighcount=zeros(1,num);%������Ԫ�صĸ���
    classsum=zeros(1,num);%��ʽ3-2�����
    classcount=zeros(1,num);%ÿһ������������������
    ss=0;%ÿ���ѡ���Ե���������Ƚ�
    tempdep=zeros(1,num);%��ʱ���������ÿһ�㶼Ҫѡ��ʱ���Ż��õ���
    
    for i=1:num
        if pos(1,i)==0%û�б���һ��ѡ��
        for k=1:num
        for jj=1:j%������ͣ��ڼ�����м�������
            d=0;
            if discreta(cporder(1,jj))==0%�����cporder(1,jj)������jjj���
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
    %��������������Ϊ0 ����Ϊȱʡֵ����1��ԭ��
    for i=1:num
        M(i,i)=0;
    end
 for i=1:num%�����������N(x) 1Ϊ��x������
     if pos(1,i)==0%û�б���һ��ѡ��
         for k=1:num
              if M(i,k)<=b
                  N(i,k)=1;
              end    
          end
     end
end   
for i=1:num%�ж��������ڵı�ǩ���
    if pos(1,i)==0%û�б���һ��ѡ��
    for k=1:num
        if N(i,k)==1
            label_s(i,trdata(k,1))=label_s(i,trdata(k,1))+1;
            neighcount(1,i)=neighcount(1,i)+1;
        end
    end
    end
end    
 %��Ҫ���һ��3-2��͵Ĺ�ʽ
for i=1:num
    if pos(1,i)==0%û�б���һ��ѡ��
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
    if pos(1,i)==0%û�б���һ��ѡ��
    for k=1:cla
        if label_s(i,k)>0
            classcount(1,i)=classcount(1,i)+1;%ÿһ������������������
        end
    end
    end
end
for i=1:num
    if pos(1,i)==0%û�б���һ��ѡ��
    if label_s(i,trdata(i,1))==neighcount(1,i)%�������Ҫ������
        c(i)=b;
    else
        c(i)=b*exp(-0.5*classsum(1,i)/nchoosek(classcount(1,i),2)/(neighcount(1,i)));%��ʽ3-2   
    end  
    end
end   
for i=1:num%�����������VN(x) 1Ϊ��x�Ŀɱ�����
    if pos(1,i)==0%û�б���һ��ѡ��
    for k=1:num
        if M(i,k)<=c(i)
            VN(i,k)=1;
        end    
    end
    end
end
for i=1:num%�����Ƿ�Ϊ����
    if pos(1,i)==0%û�б���һ��ѡ��
    for k=1:num
        if VN(i,k)==1&&trdata(i,1)~=trdata(k,1)
            tempdep(1,i)=1;%1Ϊ������
        end    
    end
    end
end
 %tempdep      ������
for i=1:num
    if pos(1,i)==0%û�б���һ��ѡ��
    if tempdep(1,i)==0
        
        ss=ss+1;
    end 
    end
end
if ss>max_s%�ж��Ƿ������ŵ�����
    max_s=ss;
    max_cord=jjj;
    for i=1:num
        if pos(1,i)==0%û�б���һ��ѡ��
            temp1dep(1,i)=1;%��1�� Ϊ1 ˵��Ϊ������
            temp1dep(1,i)=tempdep(1,i);%�ҵ����򲢵���0
        end
    end
end

        end%��Ӧ��if cp��1��i��==0
    end
    %o
    %max_s
    if max_cord==0 || max_s<theta*num%��һ��û��������ʵ���н���ֹͣ��
        cporder(1,j)=0;
        break %�����۵���  
    else
         cporder(1,j)=max_cord;%��max_cord����Ϊ���ŵ�����
         cp(1,max_cord)=1;
    end
    
for i=1:num%��һ�����ŵ�һ�����ԡ�
    if pos(1,i)==0%û�б���һ��ѡ��
    if temp1dep(1,i)==0
        
        s=s+1;
        pos(1,i)=1;%��ʱi����Ҫ������һ��������.
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
dependent(o,f)=s/num;      %������










if s<=3
    s

    ACC
    
end


    
    
    %����knn�ó����ྫȷ
    K=3;%��ʾ5NN
    TM=zeros(m-num,s);%ѵ��������Լ�֮��ľ���
    knnlabel=zeros(m-num,K);%��������������ǩ
    re=zeros(1,m-num);%���knn��Ľ��
    for i=1:m-num
        for k=1:s
            for j=1:n-1%jΪ����
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
    %������Ѱ�������k����
    sortTM=TM';%�б�ʾ����Ԫ�أ��б�ʾѵ��Ԫ��
    [Y,I]=sort(sortTM);
    for i=1:m-num
        for k=1:K
            knnlabel(i,k)=posdata(I(k,i),1);
        end
    end
    for i=1:m-num
        re(1,i)=mode(knnlabel(i,1:K));
        if re(1,i)==tsdata(i,1)
            accracy(1,o)=accracy(1,o)+1;%+1��ʾ����һ��
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