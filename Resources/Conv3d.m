function param=Conv3d(image,W,b)
  filtersize=size(W,1);
  inchannel=size(W,3);
  outchannel=size(W,4);
  %numImage=size(iamge,4);
  imagedim=size(image,1);
  convdim=imagedim-filtersize+1;
  param=zeros(convdim,convdim,outchannel);
  
  for fil2=1:outchannel
      outimage=zeros(convdim,convdim);
      for fil1=1:inchannel
          filter=squeeze(W(:,:,fil1,fil2));
          filter=rot90(squeeze(filter),2);
          tmpimg=squeeze(image(:,:,fil1));
          outimage=outimage+conv2(tmpimg,filter,'valid');
      end
      outimage=bsxfun(@plus,outimage,b(fil2));
      outimage=1./(1+exp(-outimage));
      param(:,:,fil2)=outimage;
  end
end
              