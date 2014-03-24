%% Display image in correct coordinate system
img = load('/home/beams/EPIX34ID/yoon/singfel/dataShrine/img1.dat','-ascii');
figure, imagesc(img), axis xy equal tight, xlabel('x'), ylabel('y'), title('img1')

img = load('/home/beams/EPIX34ID/yoon/singfel/dataShrine/img1.dat','-ascii');
figure, imagesc(log(abs(img))), axis xy equal tight, xlabel('x'), ylabel('y')
img = load('/home/beams/EPIX34ID/yoon/singfel/dataShrine/img2.dat','-ascii');
figure, imagesc(log(abs(img))), axis xy equal tight, xlabel('x'), ylabel('y')
img = load('/home/beams/EPIX34ID/yoon/singfel/dataShrine/img3.dat','-ascii');
figure, imagesc(log(abs(img))), axis xy equal tight, xlabel('x'), ylabel('y')
%% Check study
myIntensity = load('/home/beams/EPIX34ID/yoon/singfel/build/myIntensityStudy.dat','-ascii');
[pix2d,mySize] = size(myIntensity);
myI = zeros(mySize,mySize,mySize);
for i = 1:mySize
    myI(:,:,i) = reshape(myIntensity(:,i),mySize,mySize);
end
figure
subplot(131), imagesc(log(abs(myI(:,:,ceil(mySize/2))))), axis xy equal tight, xlabel('x'), ylabel('y'), title('Z plane')
subplot(132), imagesc(log(abs(squeeze(myI(:,ceil(mySize/2),:))))), axis xy equal tight, xlabel('x'), ylabel('z'), title('Y plane')
subplot(133), imagesc(log(abs(squeeze(myI(ceil(mySize/2),:,:))))), axis xy equal tight, xlabel('y'), ylabel('z'), title('X plane')

%% Check
load('/home/beams/EPIX34ID/yoon/singfel/build/myIntensity.dat','-ascii');
[pix2d,mySize] = size(myIntensity);
myI = zeros(mySize,mySize,mySize);
for i = 1:mySize
    myI(:,:,i) = reshape(myIntensity(:,i),mySize,mySize);
end
figure
subplot(131), imagesc(log(abs(myI(:,:,ceil(mySize/2))))), axis image, title('XY slice')
subplot(132), imagesc(log(abs(squeeze(myI(:,ceil(mySize/2),:))))), axis image, title('XZ slice')
subplot(133), imagesc(log(abs(squeeze(myI(ceil(mySize/2),:,:))))), axis image, title('YZ slice')
%%
figure, imagesc(reshape(log(abs(myIntensity(:,ceil(mySize/2)))),mySize,mySize)), axis image
figure
for i = 1:mySize
    imagesc(reshape(log(abs(myIntensity(:,i))),mySize,mySize)), axis image, title(i), caxis([-10 15]), drawnow
    pause(0.1)
end
figure, imagesc(reshape(log(abs(myIntensity(:,60))),mySize,mySize)), axis image
figure
for i = 1:mySize
    imagesc(log(abs(squeeze(myI(i,:,:))))), axis image, drawnow
    pause(0.1)
end
%%
figure
p2 = patch(isosurface(myI,1),...
    'FaceColor','red','EdgeColor','none','FaceAlpha',0.5);
isonormals(myI,p2)
view(3); daspect([1 1 1]); axis tight
camlight;  camlight(-80,-10); lighting phong;
title('Data Normals')
