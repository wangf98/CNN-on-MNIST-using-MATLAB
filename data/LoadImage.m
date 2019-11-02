function image=LoadImage(filename)
  fp=fopen(filename, 'rb');
  magicnum=fread(fp,1,'int32',0,'ieee-be');
  numimage=fread(fp,1,'int32',0,'ieee-be');
  numrow=fread(fp,1,'int32',0,'ieee-be');
  numcol=fread(fp,1,'int32',0,'ieee-be');
  
  image=fread(fp,inf,'unsigned char');
  image=reshape(image,numcol,numrow,numimage);
  image=permute(image,[2 1 3]);
  
  fclose(fp);
  image=reshape(image,size(image,1),size(image,2),size(image,3));
  image=double(image)/255;
end