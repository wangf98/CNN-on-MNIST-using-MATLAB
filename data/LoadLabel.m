function label=LoadLabel(filename)
  fp=fopen(filename,'rb');
  magicnum=fread(fp,1,'int32',0,'ieee-be');
  numlabel=fread(fp,1,'int32',0,'ieee-be');
  label=fread(fp,inf,'unsigned char');
  fclose(fp);
end