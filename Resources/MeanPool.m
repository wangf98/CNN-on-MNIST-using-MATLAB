function param=MeanPool(pooldim, W)
  %numimg=size(W,4);
  numchannel=size(W,3);
  convdim=size(W,1);
  param=zeros(convdim/pooldim,convdim/pooldim,numchannel);
  for j=1:numchannel
      originalparam=squeeze(W(:,:,j));
      tempparam=conv2(originalparam,ones(pooldim)/(pooldim^2),'valid');
      param(:,:,j)=tempparam(1:pooldim:end,1:pooldim:end);
  end
end