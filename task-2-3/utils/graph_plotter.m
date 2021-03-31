clc;
clear all;
close all;

% getting data from csv files
enet= csvread('C:\Users\Shayari B\Desktop\NNTI RESULTS\NNTI_CSV_Results_ENet.csv');
r2= csvread('C:\Users\Shayari B\Desktop\NNTI RESULTS\NNTI_CSV_Results_R2UNet.csv');
psp=csvread('C:\Users\Shayari B\Desktop\NNTI RESULTS\NNTI_CSV_Results_PSPNet.csv');

%getting parameters for ENET
enet_dice= enet(:, 1);
enet_f1=enet(:, 2);
enet_lr=enet(:, 3);
enet_miou=enet(:, 4);
enet_pa= enet(:, 5);
enet_se=enet(:, 6);
enet_loss= enet(:, 7);

enet_val_dice= enet(:, 8);
enet_val_f1= enet(:, 9);
enet_val_miou= enet(:, 10);
enet_val_pa=enet(:, 11);
enet_val_se=enet(:, 12);
enet_val_loss= enet(:, 13);

%getting parameters for R2Unet
r2_dice= r2(:, 1);
r2_f1=r2(:, 2);
r2_lr=r2(:, 3);
r2_miou=r2(:, 4);
r2_pa= r2(:, 5);
r2_se=r2(:, 6);
r2_loss= r2(:, 7);
size(r2_loss)
r2_val_dice= r2(:, 8);
size(r2_val_dice)
r2_val_f1= r2(:, 9);
size(r2_val_f1)
r2_val_miou= r2(:, 10);
r2_val_pa=r2(:, 11);
r2_val_se=r2(:, 12);
r2_val_loss= r2(:, 13);

%getting parameters for PSPnet
psp_dice= psp(:, 1);
psp_f1=psp(:, 2);
psp_lr=psp(:, 3);
psp_miou=psp(:, 4);
psp_pa= psp(:, 5);
psp_se=psp(:, 6);
psp_loss= psp(:, 7);

psp_val_dice= psp(:, 8);
psp_val_f1= psp(:, 9);
psp_val_miou= psp(:, 10);
psp_val_pa=psp(:, 11);
psp_val_se=psp(:, 12);
psp_val_loss= psp(:, 13);




%Plot generation for ENET

figure(1)
plot(1:80,enet_dice(1:80),'LineWidth',2)
grid on
xlabel('Epochs','FontSize',24)
ylabel('Dice Coeffient','FontSize',24)
title('ENet - Dice Coefficient','FontSize',24)

figure(2)
plot(1:80,enet_f1(1:80),'LineWidth',2)
grid on
xlabel('Epochs','FontSize',24)
ylabel('F1-Score','FontSize',24)
title('ENet - F1-Score','FontSize',24)

figure(3)
plot(1:80,enet_lr(1:80),'LineWidth',2)
grid on
xlabel('Epochs','FontSize',24)
ylabel('Learning Rate','FontSize',24)
title('ENet - Learning Rate','FontSize',24)

figure(4)
plot(1:80,enet_miou(1:80),'LineWidth',2)
grid on
xlabel('Epochs','FontSize',24)
ylabel('Mean IoU','FontSize',24)
title('ENet - Mean IoU','FontSize',24)

figure(5)
plot(1:80,enet_pa(1:80),'LineWidth',2)
grid on
xlabel('Epochs','FontSize',24)
ylabel('Pixel Accuracy','FontSize',24)
title('ENet - Pixel Accuracy','FontSize',24)

figure(6)
plot(1:80,enet_se(1:80),'LineWidth',2)
grid on
xlabel('Epochs','FontSize',24)
ylabel('Sensitivity','FontSize',24)
title('ENet - Sensitivity','FontSize',24)

figure(7)
plot(1:length(enet_loss),enet_loss,'LineWidth',2)
grid on
xlabel('Epochs','FontSize',24)
ylabel('loss','FontSize',24)
title('ENet - Loss','FontSize',24)

figure(8)
plot(1:15,enet_val_dice(1:15),'LineWidth',2)
grid on
xlabel('Epochs','FontSize',24)
ylabel('Dice Coeffient','FontSize',24)
title('ENet - Validation - Dice Coefficient','FontSize',24)

figure(9)
plot(1:15,enet_val_f1(1:15),'LineWidth',2)
grid on
xlabel('Epochs','FontSize',24)
ylabel('F1-Score','FontSize',24)
title('ENet - Validation - F1-Score','FontSize',24)

figure(10)
plot(1:15,enet_val_miou(1:15),'LineWidth',2)
grid on
xlabel('Epochs','FontSize',24)
ylabel('Mean IoU','FontSize',24)
title('ENet - Validation - Mean IoU','FontSize',24)

figure(11)
plot(1:15,enet_val_pa(1:15),'LineWidth',2)
grid on
xlabel('Epochs','FontSize',24)
ylabel('Pixel Accuracy','FontSize',24)
title('ENet - Validation - Pixel Accuracy','FontSize',24)

figure(12)
plot(1:15,enet_val_se(1:15),'LineWidth',2)
grid on
xlabel('Epochs','FontSize',24)
ylabel('Sensitivity','FontSize',24)
title('ENet - Validation - Sensitivity','FontSize',24)

figure(13)
plot(1:15,enet_val_loss(1:15),'LineWidth',2)
grid on
xlabel('Epochs','FontSize',24)
ylabel('loss','FontSize',24)
title('ENet - Validation - Loss','FontSize',24)


%Plot Generation for R2UNET and PSPNET
figure(14)
plot(1:80,r2_dice(1:80),'LineWidth',2)
hold on
plot(1:80,psp_dice(1:80),'LineWidth',2)
grid on
legend('R2UNet','PSPNet','FontSize',22)
xlabel('Epochs','FontSize',24)
ylabel('Dice Coeffient','FontSize',24)
title('Dice Coefficient','FontSize',24)

figure(15)
plot(1:80,r2_f1(1:80),'LineWidth',2)
hold on
plot(1:80,psp_f1(1:80),'LineWidth',2)
grid on
legend('R2UNet','PSPNet','FontSize',22)
xlabel('Epochs','FontSize',24)
ylabel('F1 Score','FontSize',24)
title('F1 Score','FontSize',24)

figure(16)
plot(1:80,r2_lr(1:80),'LineWidth',2)
hold on
plot(1:80,psp_lr(1:80),'LineWidth',2)
grid on
legend('R2UNet','PSPNet','FontSize',22)
xlabel('Epochs','FontSize',24)
ylabel('Learning rate','FontSize',24)
title('Learning Rate','FontSize',24)

figure(17)
plot(1:80,r2_miou(1:80),'LineWidth',2)
hold on
plot(1:80,psp_miou(1:80),'LineWidth',2)
grid on
legend('R2UNet','PSPNet','FontSize',22)
xlabel('Epochs','FontSize',24)
ylabel('Mean IoU','FontSize',24)
title('Mean IoU','FontSize',24)

figure(18)
plot(1:80,r2_pa(1:80),'LineWidth',2)
hold on
plot(1:80,psp_pa(1:80),'LineWidth',2)
grid on
legend('R2UNet','PSPNet','FontSize',22)
xlabel('Epochs','FontSize',24)
ylabel('Pixel Accuracy','FontSize',24)
title(' Pixel Accuracy','FontSize',24)

figure(19)
plot(1:80,r2_se(1:80),'LineWidth',2)
hold on
plot(1:80,psp_se(1:80),'LineWidth',2)
grid on
legend('R2UNet','PSPNet','FontSize',22)
xlabel('Epochs','FontSize',24)
ylabel('Sensitivity','FontSize',24)
title('Sensitivity','FontSize',24)

figure(20)
plot(1:1000,r2_loss,'LineWidth',2)
hold on
plot(1:1000,psp_loss,'LineWidth',2)
grid on
legend('R2UNet','PSPNet','FontSize',22)
xlabel('Epochs','FontSize',24)
ylabel('loss','FontSize',24)
title('Loss','FontSize',24)

figure(21)
plot(1:15,r2_val_dice(1:15),'LineWidth',2)
hold on
plot(1:15,psp_val_dice(1:15),'LineWidth',2)
grid on
legend('R2UNet','PSPNet','FontSize',22)
xlabel('Epochs','FontSize',24)
ylabel('Dice Coeffient','FontSize',24)
title('Validation - Dice Coefficient','FontSize',24)

figure(22)
plot(1:15,r2_val_f1(1:15),'LineWidth',2)
hold on
plot(1:15,psp_val_f1(1:15),'LineWidth',2)
grid on
legend('R2UNet','PSPNet','FontSize',22)
xlabel('Epochs','FontSize',24)
ylabel('F1-Score','FontSize',24)
title('Validation - F1-Score','FontSize',24)

figure(23)
plot(1:15,r2_val_miou(1:15),'LineWidth',2)
hold on
plot(1:15,psp_val_miou(1:15),'LineWidth',2)
grid on
legend('R2UNet','PSPNet','FontSize',22)
xlabel('Epochs','FontSize',24)
ylabel('Mean IoU','FontSize',24)
title('Validation - Mean IoU','FontSize',24)

figure(24)
plot(1:15,r2_val_pa(1:15),'LineWidth',2)
hold on
plot(1:15,psp_val_pa(1:15),'LineWidth',2)
grid on
legend('R2UNet','PSPNet','FontSize',22)
xlabel('Epochs','FontSize',24)
ylabel('Pixel Accuracy','FontSize',24)
title('Validation - Pixel Accuracy','FontSize',24)

figure(25)
plot(1:15,r2_val_se(1:15),'LineWidth',2)
hold on
plot(1:15,psp_val_se(1:15),'LineWidth',2)
grid on
legend('R2UNet','PSPNet','FontSize',22)
xlabel('Epochs','FontSize',24)
ylabel('Sensitivity','FontSize',24)
title('Validation - Sensitivity','FontSize',24)

figure(26)
plot(1:15,r2_val_loss(1:15),'LineWidth',2)
hold on
plot(1:15,psp_val_loss(1:15),'LineWidth',2)
grid on
legend('R2UNet','PSPNet','FontSize',22)
xlabel('Epochs','FontSize',24)
ylabel('loss','FontSize',24)
title('Validation - Loss','FontSize',24)








