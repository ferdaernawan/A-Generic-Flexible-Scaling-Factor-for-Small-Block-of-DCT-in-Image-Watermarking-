clear;close all;
clc
t1=cputime;
%    A= imread('lena.tiff'); 
%    A= imread('sailboat.tiff'); 
%    A= imread('baboon.tiff');
%    A= imread('pepper.tiff');
    A= imread('house.bmp');
%    A= imread('airplane.tiff');
%    A= imread('Boat.tiff');
%    A= imread('barbara.tiff');
%     A= imread('Tiffany.tiff');
%     A= imread('splash.tiff');
step=0;
% step = step + 1;
% figure(step);
% imshow(A);
% B=double(A);

B=double(A);

%YUV color matrix
% Y =  0.2990*B(:,:,1) + 0.5870*B(:,:,2) + 0.1140*B(:,:,3);
% Cr=  0.5000*B(:,:,1) - 0.4187*B(:,:,2) - 0.0813*B(:,:,3)+128;
% Cb= -0.1687*B(:,:,1) - 0.3313*B(:,:,2) + 0.5000*B(:,:,3)+128;
% 
% [M,N]=size(Y);

Y =  0.2990*B(:,:,1) + 0.5870*B(:,:,2) + 0.1140*B(:,:,3);
Cr=  0.5000*B(:,:,1) - 0.4187*B(:,:,2) - 0.0813*B(:,:,3)+128;
Cb= -0.1687*B(:,:,1) - 0.3313*B(:,:,2) + 0.5000*B(:,:,3)+128;

% % Converting RGB to YCoCg-R
% Co =  B(:,:,1)-B(:,:,3);
% Tx=B(:,:,3)+(Co./2);
% Cg=  B(:,:,2) - Tx;
% Y= Tx+(Cg./2);

[M,N]=size(Y);
% step = step + 1;
% figure(step);
% imshow(uint8(Y));

% Selecting Y Component
% [M,N]=size(Y);


%% Read watermark

OriWatermark=imread('wt1.bmp');
OriWatermark=im2bw(OriWatermark, 0.6);
[Mc Nc] = size(OriWatermark);size_watermark=Mc*Nc;
img_wat=double(OriWatermark);
% imwrite(OriWatermark,'watermark B.tiff','tiff');
%Generate key as an image
key = keyGen(Mc*Nc);
password=reshape(key(1:Nc*Mc),Nc,Mc);
step = step + 1;figure(step);imshow(password);title('Key Generation');


%The watermark is encrypted before it used in watermarking
EncImg = imageProcess(img_wat,key);
step = step + 1;figure(step);imshow(EncImg);title('Encrypted Watermark image');
imwrite(EncImg,'Encoded.tiff','tiff');
%EncImg = double(img_wat);
% --------reshape watermark size
message=reshape(EncImg,size_watermark,1,1);


%% Calculating complexity block using variance Function
y=1;x=1;u=1;
blocksize=4;
Numblock=(M/blocksize)*(M/blocksize); %--4096 block
OriY=uint8(Y);
OriY2=double(OriY);
for (ll = 1:Numblock)
   BlockOriY2=OriY2(y:y+blocksize-1,x:x+blocksize-1);
   BlockOriY2=dct2(BlockOriY2);
   V = var(BlockOriY2);
   Variance(1,ll)= var(V);
   Variance(2,ll)=x;
   Variance(3,ll)=y;
    if (x+blocksize) >= N
         x=1;
         y=y+blocksize;
     else
         x=x+blocksize;
    end
 end
Variance=Variance';
VarianceSort=sortrows(Variance);

%% Selected DCT coefficients
t1=3;
t2=1;
k=1;
%% Embedding process


%% 

A2 = uint8(OriY2);
% step = step + 1;
% figure(step);
% imshow(A2);
title('Watermarked Image');
file_name='The watermarked image.tiff';
imwrite(A2,file_name,'tiff');

step = step + 1;
figure(step);
imshow(uint8(Y));

[M N]=size(A2)
FullError= sum(sum(abs(double(A2)-double(Y))))/(M*N)
R=double(Y)-double(A2);
MSET=sum(sum(R.^2))/(M*N)
PSNRT=10*log10(255^2/MSET)
K = [0.01 0.03];
window = fspecial('gaussian', 11, 1.5);
L = 255;
SSIMT=ssim(double(A2),double(Y),K,window,L)
msgx=sprintf('\n\n-------------------------\nWatermarked Image by DCT ARE=%.3f PSNR=%.3fdB  SSIM=%.3f  \n-----------------------------\n\n',FullError, PSNRT, SSIMT);
disp(msgx);

file=imread('The watermarked image.tiff');  
R=3;
%No attacks
step = step + 1;figure(step); imshow(uint8(file));title('The Watermarked Image without attacks');
imwrite(uint8(file),'1.tiff','tiff');

%########Image Processing Attacks##############
%% Filter Attacks
%% Gaussian Lowpass Filter
 GLF=fspecial('gaussian',3,0.5);
 GLF1=filter2(GLF,file);
 B2 =zeros(M,N);
 B2 = GLF1;
 step = step + 1;figure(step); imshow(uint8(B2));title('The Watermarked Image with Gaussian Lowpass Filter');
 imwrite(uint8(B2),'2.tiff','tiff');
%% Median Filter
MF1= medfilt2(file,[3, 3]);
B2 =zeros(M,N);
B2 = MF1;
step = step + 1;figure(step); imshow(uint8(B2));title('The Watermarked Image with Median Filter');
imwrite(uint8(B2),'3.tiff','tiff');
%% Average Filter
GLF=fspecial('average',[3 3]);
GLF1=filter2(GLF,file);
B1 =zeros(M,N);
B1 = GLF1;
step = step + 1;figure(step); imshow(uint8(B1));title('The Watermarked Image with Gaussian Lowpass Filter');
imwrite(uint8(B1),'4.tiff','tiff');
%% wiener Filter     
AW = wiener2(file,[3 3]);
step = step + 1;figure(step); imshow(uint8(AW));title('The Watermarked Image with Wiener Filter');
imwrite(uint8(AW),'5.tiff','tiff');

%% Gaussian noise  
GN1=imnoise(file,'gaussian',0,0.001);
step = step + 1;figure(step); imshow(uint8(GN1));title('The Watermarked Image with Gaussian Noise');
imwrite(uint8(GN1),'6.tiff','tiff');
%Gaussian Noise    
GN1=imnoise(file,'gaussian',0,0.002);
step = step + 1;figure(step); imshow(uint8(GN1));title('The Watermarked Image with Gaussian Noise');
imwrite(uint8(GN1),'7.tiff','tiff');
%Gaussian Noise    
 GN1=imnoise(file,'gaussian',0,0.003);
 step = step + 1;figure(step); imshow(uint8(GN1));title('The Watermarked Image with Gaussian Noise');
 imwrite(uint8(GN1),'8.tiff','tiff');

%% Salt and Pepper noise
SP=imnoise(file,'salt & pepper',0.001);
step = step + 1;figure(step); imshow(uint8(SP));title('The Watermarked Image with Salt Pepper Noise');
imwrite(uint8(SP),'9.tiff','tiff');
% Salt and Pepper noise
SP=imnoise(file,'salt & pepper',0.002);
step = step + 1;figure(step); imshow(uint8(SP));title('The Watermarked Image with Salt Pepper Noise');
imwrite(uint8(SP),'10.tiff','tiff');
%Salt and Pepper noise
SP=imnoise(file,'salt & pepper',0.005);
step = step + 1;figure(step); imshow(uint8(SP));title('The Watermarked Image with Salt Pepper Noise');
imwrite(uint8(SP),'11.tiff','tiff');
%Salt and Pepper noise
SP=imnoise(file,'salt & pepper',0.01);
step = step + 1;figure(step); imshow(uint8(SP));title('The Watermarked Image with Salt Pepper Noise');
imwrite(uint8(SP),'12.tiff','tiff');
%Salt and Pepper noise
SP=imnoise(file,'salt & pepper',0.02);
step = step + 1;figure(step); imshow(uint8(SP));title('The Watermarked Image with Salt Pepper Noise');
imwrite(uint8(SP),'13.tiff','tiff');
%Salt and Pepper noise
SP=imnoise(file,'salt & pepper',0.03);
step = step + 1;figure(step); imshow(uint8(SP));title('The Watermarked Image with Salt Pepper Noise');
imwrite(uint8(SP),'14.tiff','tiff');
%% Speckle noise
%Speckle noise
SPCK= imnoise(file,'speckle',.001);
step = step + 1;figure(step); imshow(uint8(SPCK));title('The Watermarked Image with Speckle noise');
imwrite(uint8(SPCK),'15.tiff','tiff');
SPCK= imnoise(file,'speckle',.003);
step = step + 1;figure(step); imshow(uint8(SPCK));title('The Watermarked Image with Speckle noise');
imwrite(uint8(SPCK),'16.tiff','tiff');
SPCK= imnoise(file,'speckle',.005);
step = step + 1;figure(step); imshow(uint8(SPCK));title('The Watermarked Image with Speckle noise');
imwrite(uint8(SPCK),'17.tiff','tiff');
SPCK= imnoise(file,'speckle',.01);
step = step + 1;figure(step); imshow(uint8(SPCK));title('The Watermarked Image with Speckle noise');
imwrite(uint8(SPCK),'18.tiff','tiff');
%% Sharpening 
HSharpening = padarray(2,[2 2]) - fspecial('gaussian' ,[5 5],2); % create unsharp mask
sharpened1 = imfilter(file,HSharpening);  % create a sharpened version of the image using that mask
B2= sharpened1;
step = step + 1;figure(step); imshow(uint8(B2));title('The Watermarked Image with Sharpening');
imwrite(uint8(B2),'19.tiff','tiff');
%%Poisson
PSSN= imnoise(file,'poisson');
step = step + 1;figure(step); imshow(uint8(PSSN));title('The Watermarked Image with Poisson');
imwrite(uint8(PSSN),'20.tiff','tiff');
%% Histogram
J=histeq(file);
step = step + 1;figure(step); imshow(uint8(J));title('The Watermarked Image with Histogram Equalized');
imwrite(uint8(J),'21.tiff','tiff');

% % %#########Geometrical Attacks#########################
%% Centred Cropping off 12.5%
 C1=file;
 CR1=imcrop(C1,[192 192 128 128]);
 [x,y]=size(CR1);
 for i=1:x
     for j=1:y
         C1(i+191,j+191)=0;
     end
 end
 step = step + 1;figure(step); imshow(uint8(C1));title('The Watermarked Image with Centred Cropping off 25%');
  imwrite(uint8(C1),'22.tiff','tiff');

%%% Centred Cropping off 50%
%   C2=file;
%  CR1=imcrop(C2,[64 64 374 374]);
%  [x,y]=size(CR1);
%  for i=1:x
%      for j=1:y
%          C2(i+63,j+63)=0;
%      end
%  end
%  step = step + 1;figure(step); imshow(uint8(C2));title('The Watermarked Image with Centred Cropping off 50%');
%  imwrite(uint8(C2),'5.tiff','tiff');
% %  
% % %% Croping off 25%
% % C3=file;
% % CR1=imcrop(C3,[64 64 512 512]);
% % [x,y]=size(CR1);
% % CRP1=zeros(M,N);
% % for i=1:x
% %     for j=1:y
% %         CRP1(i+63,j+63)=CR1(i,j);
% %     end
% % end
% % step = step + 1;figure(step); imshow(uint8(CRP1));title('The Watermarked Image with Croping off 25%');
% % imwrite(uint8(CRP1),'16.tiff','tiff');
% % 
% % %% Croping off 50%
% % CR1=imcrop(file,[128 128 512 512]);
% % [x,y]=size(CR1);
% % CRP1=zeros(M,N);
% % for i=1:x
% %     for j=1:y
% %         CRP1(i+127,j+127)=CR1(i,j);
% %     end
% % end
% % step = step + 1;figure(step); imshow(uint8(CRP1));title('The Watermarked Image with Croping off 50%');
% % imwrite(uint8(CRP1),'17.tiff','tiff');
% % 
% % %% Croping row off 25%
% % CR1=imcrop(file,[0 128 512 512]);
% % [x,y]=size(CR1);
% % CRP1=zeros(M,N);
% % for i=1:x
% %     for j=1:y
% %         CRP1(127+i,j)=CR1(i,j);
% %     end
% % end
% % step = step + 1;figure(step); imshow(uint8(CRP1));title('The Watermarked Image with Croping row off 25%');
% % imwrite(uint8(CRP1),'18.tiff','tiff');
% % 
% % %% Croping row off 50%
% % CR1=imcrop(file,[0 256 512 512]);
% % [x,y]=size(CR1);
% % CRP1=zeros(M,N);
% % for i=1:x
% %     for j=1:y
% %         CRP1(255+i,j)=CR1(i,j);
% %     end
% % end
% % step = step + 1;figure(step); imshow(uint8(CRP1));title('The Watermarked Image with Croping row off 50%');
% % imwrite(uint8(CRP1),'19.tiff','tiff');
% % 
% % %% Croping column off 25%
% % CR1=imcrop(file,[128 0 512 512]);
% % [x,y]=size(CR1);
% % CRP1=zeros(M,N);
% % for i=1:x
% %     for j=1:y
% %         CRP1(i,j+127)=CR1(i,j);
% %     end
% % end
% % step = step + 1;figure(step); imshow(uint8(CRP1));title('The Watermarked Image with Croping column off 25%');
% % imwrite(uint8(CRP1),'20.tiff','tiff');
% % 
% % %% Croping column off 50%
% % CR1=imcrop(file,[256 0 512 512]);
% % [x,y]=size(CR1);
% % CRP1=zeros(M,N);
% % for i=1:x
% %     for j=1:y
% %        CRP1(i,j+255)=CR1(i,j);
% %     end
% % end
% % step = step + 1;figure(step); imshow(uint8(CRP1));title('The Watermarked Image with Croping column off 50%');
% % imwrite(uint8(CRP1),'21.tiff','tiff');
% % 
% % %% Rotation
% % 
% % IR2=imrotate(file,45,'crop');
% % step = step + 1;figure(step); imshow(uint8(IR2));title('The Watermarked Image with Rotation 8');
% % imwrite(uint8(IR2),'22.tiff','tiff');
% % 
% % IR3=imrotate(file,70,'crop');
% % step = step + 1;figure(step); imshow(uint8(IR3));title('The Watermarked Image with Rotation 15');
% % imwrite(uint8(IR3),'23.tiff','tiff');
% % 
% % %% Translate Attack 
% % 
% % K1 = imtranslate(file,[10, 10]);
% % step = step + 1;figure(step); imshow(uint8(K1));title('The Watermarked Image with Translated Image 10 10');
% % imwrite(uint8(K1),'24.tiff','tiff');
% % 
% % K2 = imtranslate(file,[10, 20]);
% % step = step + 1;figure(step); imshow(uint8(K2));title('The Watermarked Image with Translated Image 10 20');
% % imwrite(uint8(K2),'25.tiff','tiff');
% %  
% % K3 = imtranslate(file,[30, 40]);
% % step = step + 1;figure(step); imshow(uint8(K3));title('The Watermarked Image with Translated Image 30 40');
% % imwrite(uint8(K3),'26.tiff','tiff');
% % 
% % 
%  %% SCALING
% %  L1=imresize(file,0.5);
% %  step = step + 1;figure(step); imshow(uint8(L1));title('The Watermarked Image with Scaling 0.5');
% %  imwrite(uint8(L1),'27.tiff','tiff');
% %  file1=imread('27.tiff');
% %  L11=imresize(file1,2);
% %  step = step + 1;figure(step); imshow(uint8(L11));title('The Watermarked Image with Scaling 0.5');
% %  imwrite(uint8(L11),'27.tiff','tiff');
% %  
% %   L2=imresize(file,0.25);
% %   step = step + 1;figure(step); imshow(uint8(L2));title('The Watermarked Image with Scaling 0.25');
% %   imwrite(uint8(L2),'28.tiff','tiff');
% %   file2=imread('28.tiff');
% %   L22=imresize(file2,4);
% %   step = step + 1;figure(step); imshow(uint8(L22));title('The Watermarked Image with Scaling 0.25');
% %   imwrite(uint8(L22),'28.tiff','tiff');
% %   
% %   L3=imresize(file,0.25);
% %   step = step + 1;figure(step); imshow(uint8(L3));title('The Watermarked Image with Scaling 0.25');
% %   imwrite(uint8(L3),'29.tiff','tiff');
% %   file2=imread('29.tiff');
% %   L33=imresize(file2,4);
% %   step = step + 1;figure(step); imshow(uint8(L33));title('The Watermarked Image with Scaling 0.25');
% %   imwrite(uint8(L33),'29.tiff','tiff');

%% ATTACK JPEG Compression
% imwrite(file,'1.jpg','jpg','quality', 10)
% A1=imread('1.jpg');
% step = step + 1;figure(step); imshow(uint8(A1));title('The Watermarked Image with JPEG Compression Q10.jpg'); 
% 
% imwrite(file,'2.jpg','jpg','quality', 20)
% A2=imread('2.jpg');
% step = step + 1;figure(step); imshow(uint8(A2));title('The Watermarked Image with JPEG Compression Q10.jpg'); 
% 
% imwrite(file,'3.jpg','jpg','quality', 30)
% A3=imread('3.jpg');
% step = step + 1;figure(step); imshow(uint8(A3));title('The Watermarked Image with JPEG Compression Q30.jpg');
% 
% imwrite(file,'4.jpg','jpg','quality', 40)
% A4=imread('4.jpg');
% step = step + 1;figure(step); imshow(uint8(A4));title('The Watermarked Image with JPEG Compression Q30.jpg');
% 
imwrite(file,'5.jpg','jpg','quality', 50)
A5=imread('5.jpg');
step = step + 1;figure(step); imshow(uint8(A5));title('The Watermarked Image with JPEG Compression Q50.jpg'); 

% imwrite(file,'6.jpg','jpg','quality', 60)
% A6=imread('6.jpg');
% step = step + 1;figure(step); imshow(uint8(A6));title('The Watermarked Image with JPEG Compression Q50.jpg'); 
% 
% imwrite(file,'7.jpg','jpg','quality', 70)
% A7=imread('7.jpg');
% step = step + 1;figure(step); imshow(uint8(A7));title('The Watermarked Image with JPEG Compression Q70.jpg'); 
% 
% imwrite(file,'8.jpg','jpg','quality', 80)
% A8=imread('8.jpg');
% step = step + 1;figure(step); imshow(uint8(A8));title('The Watermarked Image with JPEG Compression Q70.jpg'); 
% 
% imwrite(file,'9.jpg','jpg','quality', 90)
% A9=imread('9.jpg');
% step = step + 1;figure(step); imshow(uint8(A9));title('The Watermarked Image with JPEG Compression Q90.jpg');


% %% ATTACK JPEG2000
% imwrite(file,'1.jp2','jp2','Mode','lossy','CompressionRatio',2);
% C1=imread('1.jp2');
% step = step + 1;figure(step); imshow(uint8(C1));title('The Watermarked Image with JPEG2000 CompressionRatio 1.jpg');
% 
% imwrite(file,'2.jp2','jp2','Mode','lossy','CompressionRatio',4);
% C2=imread('2.jp2');
% step = step + 1;figure(step); imshow(uint8(C2));title('The Watermarked Image with JPEG2000 CompressionRatio 2.jpg');
% 
% imwrite(file,'3.jp2','jp2','Mode','lossy','CompressionRatio',6);
% C3=imread('3.jp2');
% step = step + 1;figure(step); imshow(uint8(C3));title('The Watermarked Image with JPEG2000 CompressionRatio 3.jpg');
% 
% imwrite(file,'4.jp2','jp2','Mode','lossy','CompressionRatio',8);
% C4=imread('4.jp2');
% step = step + 1;figure(step); imshow(uint8(C4));title('The Watermarked Image with JPEG2000 CompressionRatio 4.jpg');
% 
% imwrite(file,'5.jp2','jp2','Mode','lossy','CompressionRatio',10);
% C5=imread('5.jp2');
% step = step + 1;figure(step); imshow(uint8(C5));title('The Watermarked Image with JPEG2000 CompressionRatio 5.jpg');
% 
% imwrite(file,'6.jp2','jp2','Mode','lossy','CompressionRatio',12);
% C6=imread('6.jp2');
% step = step + 1;figure(step); imshow(uint8(C6));title('The Watermarked Image with JPEG2000 CompressionRatio 6.jpg');
% 
% imwrite(file,'7.jp2','jp2','Mode','lossy','CompressionRatio',14);
% C7=imread('7.jp2');
% step = step + 1;figure(step); imshow(uint8(C7));title('The Watermarked Image with JPEG2000 CompressionRatio 7.jpg');
% 
% imwrite(file,'8.jp2','jp2','Mode','lossy','CompressionRatio',16);
% C8=imread('8.jp2');
% step = step + 1;figure(step); imshow(uint8(C8));title('The Watermarked Image with JPEG2000 CompressionRatio 8.jpg');
% 
% imwrite(file,'9.jp2','jp2','Mode','lossy','CompressionRatio',18);
% C9=imread('9.jp2');
% step = step + 1;figure(step); imshow(uint8(C9));title('The Watermarked Image with JPEG2000 CompressionRatio 9.jpg');

%EXTRACTION PART



% Watermark=Watermark';
Watermark=reshape(Watermark(1:size_watermark),Mc,Nc);
Watermark=uint8(Watermark);
Watermark=double(Watermark);
%message2 = Watermark;                     % no encryption
message2 = imageProcess(Watermark,key);  %using encryption

step = step + 1;figure(step);
imshow(message2);title('Recovery Watermark');
file_name=['Recovered Watermark IP attack no', num2str(zz),'.tiff'];
imwrite(message2,file_name,'tiff');
P=img_wat.*message2;
Q=sum(sum(P)); 
O=sqrt(sum(sum(img_wat.^2)))*sqrt(sum(sum(message2.^2)));
NC=Q/O;
[K,L]=size(img_wat);
BCR=1-sum(xor(img_wat(:),message2(:)))/(K*L);
BitError=Biter(img_wat,message2);
msg2=sprintf('\n-------------------------\nWatermark by  attack no %d  NC=%.3f BCR=%.3f BER=%.3f\n-----------------------------\n',zz, NC,BCR,BitError);
disp(msg2);
DataNC2(zz,1)=NC;
DataNC2(zz,2)=BitError;
%  xlswrite('DCTimageattackJPEG.xlsx', NC,['Sheet',num2str(1)],['A',num2str(zz)]);
%  xlswrite('DCTimageattackJPEG.xlsx', BCR,['Sheet',num2str(1)],['B',num2str(zz)]);
%  xlswrite('DCTimageattackJPEG.xlsx', BitError,['Sheet',num2str(1)],['C',num2str(zz)]);
end


a=1;b=1;
for(zz=5:5)  
    file_name=[num2str(zz),'.jpg'];
    I=imread(file_name,'jpg');
B2=double(I);
% B2=uint8(I);

y=1;x=1;u=1;
% Numblock2=size_watermark;
Numblock2=((M/blocksize)*(M/blocksize))/4; 
yy=1;gg=1;a=1;b=1;c=1;
for (i = 1 :Numblock2)
    x=VarianceSort(i,2);
    y=VarianceSort(i,3);
    BlockF= B2(y:y+blocksize-1,x:x+blocksize-1);
    BlockF=dct2(BlockF);
    
    MeanF1c(i)= mean2(BlockF);
    
    IFF3(i)= (BlockF(t1,t2)/16)+MeanF1c(i);

    alpha2(i)= (MeanF1c(i)+IFF3(i));
    
    Rx3(i) = MeanF1c(i) / alpha2(i);
 
    DCTcoefficients(i,1) = BlockF(t1,t2);

    for ii = 1:k
   
      if((u<=Mc*Nc) && (i==location(u,1)) && (ii==location(u,2)) )
 
            if (DCTcoefficients(i,ii)<Rx3(i)) 
                  Watermark(u)=1;  
                  
                    locationX(u,1)=i;
                    locationX(u,2)=ii;
                     u=u+1;  
                   a=a+1;
             else
                   Watermark(u)=0;
                    locationX(u,1)=i;
                    locationX(u,2)=ii;
                    u=u+1;
                   c=c+1;
             end
      end
      gg=gg+1;

    end
    if (x+blocksize) >= N
         x=1;
         y=y+blocksize;
     else
         x=x+blocksize;
    end
  
end

% Watermark=Watermark';
Watermark=reshape(Watermark(1:size_watermark),Mc,Nc);
Watermark=uint8(Watermark);
Watermark=double(Watermark);
%message2 = Watermark;                     % no encryption
message2 = imageProcess(Watermark,key);  %using encryption

step = step + 1;figure(step);
imshow(message2);title('Recovery Watermark');
% step = step + 1;figure(step);
% imshow(message2);title('Recovery Watermark');
% file_name=['Recovered Watermark compare metha attack no', num2str(zz),'.tiff'];
imwrite(message2,file_name,'tiff');
P=img_wat.*message2;
Q=sum(sum(P)); 
O=sqrt(sum(sum(img_wat.^2)))*sqrt(sum(sum(message2.^2)));
NC=Q/O;
[K,L]=size(img_wat);
BCR=1-sum(xor(img_wat(:),message2(:)))/(K*L);
BitError=Biter(img_wat,message2);
msg2=sprintf('\n-------------------------\nWatermark by  attack no %d  NC=%.3f BCR=%.3f BER=%.3f\n-----------------------------\n',zz, NC,BCR,BitError);
disp(msg2);
%  xlswrite('DCT_image_comparemethapepper.xlsx', NC,['Sheet',num2str(1)],['A',num2str(zz)]);
%  xlswrite('DCT_image_comparemethapepper.xlsx', BCR,['Sheet',num2str(1)],['B',num2str(zz)]);
%  xlswrite('DCT_image_comparemethapepper.xlsx', BitError,['Sheet',num2str(1)],['C',num2str(zz)]);
end
    







