clc;clear all; close all;
% data
noise_level = [40:10:100];
ratio30 =  [87.3 94.7 96.5 95.1 97.9 96.5 96.8; % TP
            87.0 92.3 96.5 97.5 97.2 96.1 97.2;% LG
            55.7 75.2 90.5 93.3 93.3 94.7 92.6 ;% DLG
            73.6 83.5 92.2 93.7 94.4 94.7 93.7 ;% LL
            75.9   86.4  93.9   94.9   95.7   95.5    95.1];
        
ratio25 = [84.5 85.2 94.4 95.1 93.7 94.7 97.9;
            81.3 87.3 93.3 97.5 96.8 95.4 95.8;
            49.0 77.3 85.9 88.0 91.5 91.5 92.6;
            68.7 77.5 86.3 91.2 92.2 92.2 93.3;
            70.9 81.8  90.0    93.0   93.6   93.5   94.9 ];
ratio20=[77.1 83.5 89.4 87.3 89.1 88.0 90.1;
         75.0 84.9 88.0 91.5 89.8 88.4 91.9;
         49.7 65.7 69.6 80.9 83.4 86.2 90.1;
         63.7 74.6 81.7 85.2 85.9 83.8 88.7;
         66.4 77.2 82.2  86.2  87.2  86.6 90.2];
ratio15 = [69.0,75.3 , 74.2  ,73.2]; 

% plot the figures
num_ratio= 4;
num_faultType = 4;
font_size =18;
marker_type ={'bo-', 'r*-.', 'ks--','b*--','gd-.','mo--','rs-' };
% legend_text ={'TP', 'LG', 'DLG', 'LL','Averaged'};
legend_text ={'TP', 'LG', 'DLG', 'LL' };
% legend_text ={ '20 %' ,'25 %','30 %' };
% legend_text = {'40 dB','50 dB', '60 dB', '70 dB' ,'80 dB', '90 dB', '100 dB'};
position_array=[0.1 0.64 0.38 0.29;
                0.57 0.64 0.38 0.29;
                0.1 0.18 0.38 0.29;
                0.57 0.18 0.38 0.29;
                0.3 0.18 0.50 0.04];
figure('units','pixels','position',[50 300 600 500]);
set(gca,'fontName','Times New Roman') 
for j=  1: num_faultType
%     h=subplot(2,2,j);
%     set(h,'Units','normalized','Position',position_array(j,:));
%     plot(noise_level,ratio15(j,:),  marker_type{j}, 'linewidth', 1.5);
%     hold on; 
    plot(noise_level,ratio20(5,:),  marker_type{1}, 'linewidth', 1.5);
%     hold on; 
    plot(noise_level,ratio25(5,:),  marker_type{2}, 'linewidth', 1.5);
%     hold on; 
    plot(noise_level,ratio30(5,:),  marker_type{j}, 'linewidth', 1.5);
    hold on; 
    xlabel({'SNR (dB)'; '(a)'},'fontname','Times New Roman','fontsize',font_size); 
    ylabel('LAR (%)','fontname','Times New Roman','fontsize',font_size); 
%     ylabel('Averaged LAR (%)','fontname','Times New Roman','fontsize',font_size); 
    xlim([noise_level(1) noise_level(end)]);
    set(gca,'XTick',noise_level);
%     title(fault_type{j},'fontsize',font_size);
end
h = legend(legend_text, 'Orientation',...
    'horizonal','Location','SouthEast','fontname','Times New Roman','fontsize',font_size);
legend('boxoff');
% set(h, 'Units','normalized','Position', position_array(5,:)); 
% figure;
% plot(ratio,cnn1,'r-*' ,'lineWidth', 2);
% hold on;
% plot(ratio,svm1, 'b-<', 'lineWidth', 2);
% hold on;
% plot(ratio,nn1, 'g-o', 'lineWidth', 2);
% xlabel('The Ratio of Measured Buses ( %)')
% ylabel('LAR of Different Classifiers (%)')
 