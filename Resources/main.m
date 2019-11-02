dimimg=28;
numclass=10;
filtersize1=5;
filtersize2=5;
imgchannel=1;
outchannel1=8;
outchannel2=32;
pooldim1=2;
pooldim2=2;
%weight_decay=1e-5;
addpath ../data/;
images=LoadImage('../data/train-images-idx3-ubyte');
images=reshape(images,dimimg,dimimg,1,[]);
labels=LoadLabel('../data/train-labels-idx1-ubyte');
labels(labels==0)=10;

W1=1e-1*randn(filtersize1,filtersize1,imgchannel,outchannel1);
W2=1e-1*randn(filtersize2,filtersize2,outchannel1,outchannel2);

outdim1=(dimimg-filtersize1+1)/pooldim1;
outdim2=(outdim1-filtersize2+1)/pooldim2;
outsize=outdim2^2*outchannel2;

r  = sqrt(6) / sqrt(numclass+outsize+1);
Wd = rand(numclass, outsize) * 2 * r - r;

b1=zeros(outchannel1,1);
b2=zeros(outchannel2,1);
bd=zeros(numclass,1);

epochs=3;
learningrate=0.01;
datasize=length(labels);

W1_velocity = zeros(size(W1));
b1_velocity = zeros(size(b1));
W2_velocity = zeros(size(W2));
b2_velocity = zeros(size(b2));
Wd_velocity = zeros(size(Wd));
bd_velocity = zeros(size(bd));

L=[];
for epoch=1:epochs
    for i=1:datasize
        imagenow=images(:,:,:,i);
        labeltrue=labels(i);
        
        W1_velocity = zeros(size(W1));
        b1_velocity = zeros(size(b1));
        W2_velocity = zeros(size(W2));
        b2_velocity = zeros(size(b2));
        Wd_velocity = zeros(size(Wd));
        bd_velocity = zeros(size(bd));
        W2_grad = zeros(size(W2));
        W1_grad = zeros(size(W1));
        Wd_grad = zeros(size(Wd));
        b2_grad = zeros(size(b2));
        b1_grad = zeros(size(b1));
        bd_grad = zeros(size(bd));
        
        convdim1=dimimg-filtersize1+1;
        tmpoutdim1=convdim1/pooldim1;
        convdim2=tmpoutdim1-filtersize2+1;
        tmpoutdim2=convdim2/pooldim2;
        
        activition1=Conv3d(imagenow,W1,b1);
        activitionpooled1=MeanPool(pooldim1,activition1);
        activition2=Conv3d(activitionpooled1,W2,b2);
        activitionpooled2=MeanPool(pooldim2,activition2);
        activitionpooled2=reshape(activitionpooled2,[],1);
        
        probability=exp(bsxfun(@plus,Wd*activitionpooled2,bd));
        sumprob=sum(probability);
        probability=bsxfun(@times,probability,1./sumprob);
        
        logp=log(probability);
        %index=sub2ind(size(logp),labeltrue);
        loss=-logp(labeltrue);
        %wloss=weight_decay/2*(sum(Wd(:).^2)+sum(W1(:).^2)+sum(W2(:).^2));
        %loss=celoss+wloss;
        
        output=zeros(size(probability));
        output(labeltrue)=1;
        deltasoftmax=(probability-output);
        t=-deltasoftmax;
        
        deltapool2=reshape(Wd'*deltasoftmax,tmpoutdim2,tmpoutdim2,outchannel2);
        
        deltaunpool2=zeros(convdim2,convdim2,outchannel2);
        for channel=1:outchannel2
            unpool=deltapool2(:,:,channel);
            deltaunpool2(:,:,channel)=kron(unpool,ones(pooldim2))./(pooldim2^2);
        end
        
        deltaconv2=deltaunpool2.*activition2.*(1-activition2);
        
        deltapool1=zeros(tmpoutdim1,tmpoutdim1,outchannel1);
        for f1=1:outchannel1
            for f2=1:outchannel2
                deltapool1(:,:,f1)=deltapool1(:,:,f1)+convn(deltaconv2(:,:,f2),W2(:,:,f1,f2),'full');
            end
        end
        
        deltaunpool1=zeros(convdim1,convdim1,outchannel1);
        for channel=1:outchannel1
            unpool=deltapool1(:,:,channel);
            deltaunpool1(:,:,channel)=kron(unpool,ones(pooldim1))./(pooldim1^2);
        end
        
        deltaconv1=deltaunpool1.*activition1.*(1-activition1);
        
        Wd_grad=deltasoftmax*activitionpooled2';
        bd_grad=sum(deltasoftmax);
        
        for channel2=1:outchannel2
            for channel1=1:outchannel1
                W2_grad(:,:,channel1,channel2)=W2_grad(:,:,channel1,channel2)+conv2(activitionpooled1(:,:,channel1),rot90(deltaconv2(:,:,channel2),2),'valid');
            end
            temp=deltaconv2(:,:,channel2);
            b2_grad(channel2)=sum(temp(:));
        end
        
        for channel1=1:outchannel1
            for channel0=1:imgchannel
                W1_grad(:,:,channel0,channel1)=W1_grad(:,:,channel0,channel1)+conv2(imagenow(:,:,channel0),rot90(deltaconv1(:,:,channel1),2),'valid');
            end
            temp=deltaconv1(:,:,channel1);
            b1_grad(channel1)=sum(temp(:));
        end
        
        Wd_velocity=Wd_velocity+learningrate*(Wd_grad);
        bd_velocity=bd_velocity+learningrate*bd_grad;
        W2_velocity=W2_velocity+learningrate*(W2_grad);
        b2_velocity=b2_velocity+learningrate*b2_grad;
        W1_velocity=W1_velocity+learningrate*(W1_grad);
        b1_velocity=b1_velocity+learningrate*b1_grad;
        
        Wd=Wd-Wd_velocity;
        bd=bd-bd_velocity;
        W2=W2-W2_velocity;
        b2=b2-b2_velocity;
        W1=W1-W1_velocity;
        b1=b1-b1_velocity;
        
        %fprintf('Epoch %d: loss: %f\n',epoch,loss);
        L(length(L)+1)=loss;
    end
    
    testimages=LoadImage('../data/t10k-images-idx3-ubyte');
    testimages=reshape(testimages,dimimg,dimimg,1,[]);
    testlabels=LoadLabel('../data/t10k-labels-idx1-ubyte');
    testlabels(testlabels==0)=10;
    correct=0;
    for i=1:length(testimages)
        testimg=testimages(:,:,:,i);
        truelabel=testlabels(i);
        activition1=Conv3d(testimg,W1,b1);
        activitionpooled1=MeanPool(pooldim1,activition1);
        activition2=Conv3d(activitionpooled1,W2,b2);
        activitionpooled2=MeanPool(pooldim2,activition2);
        activitionpooled2=reshape(activitionpooled2,[],1);
        probability=exp(bsxfun(@plus,Wd*activitionpooled2,bd));
        sumprob=sum(probability);
        probability=bsxfun(@times,probability,1./sumprob);
        [~,pred]=max(probability);
        if pred==truelabel 
            correct=correct+1;
        end
    end
    acc=correct/length(testimages);
    fprintf('Accuracy %f\n',acc);
    plot(L);
end
        