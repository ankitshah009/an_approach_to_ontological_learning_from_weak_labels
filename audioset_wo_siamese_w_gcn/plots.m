clear

class_1 = {'Acoustic environment', 'Alarm', 'Bell', 'Deformable shell', 'Digestive', ...
 'Domestic animals, pets', 'Domestic sounds, home sounds', 'Engine', ...
 'Explosion', 'Fire', 'Generic impact sounds', 'Glass', 'Hands', ...
 'Heart sounds, heartbeat', 'Human group actions', 'Human locomotion', ...
 'Human voice', 'Liquid', 'Livestock, farm animals, working animals', ...
 'Mechanisms', 'Miscellaneous sources', 'Music genre', 'Music mood', ... 
 'Music role', 'Musical concepts', 'Musical instrument', 'Noise', ...
 'Onomatopoeia', 'Other sourceless', ... 
 'Respiratory sounds', 'Silence', 'Sound reproduction', ...
 'Specific impact sounds', 'Surface contact', 'Thunderstorm', 'Tools', ...
 'Vehicle', 'Water', 'Whistling', 'Wild animals', 'Wind', 'Wood'};
class_2 = {'Animal', 'Channel, environment and background', 'Human sounds', 'Music', ...
 'Natural sounds', 'Sounds of things', 'Source-ambiguous sounds'};
Y1 = [1583.  944.  413.  179.  406.  655. 1845.  560.  545.  122.  459.  171. ...
  170.  110.  384.  180. 7210.  722.  774.  674.  477. 2391.  411.  508. ... 
   77. 3347.  644. 1163.  270.   555.   60.  180.   61.  325.   73. ...
  393. 1960.  708.   61. 1197.  230.  280.];
Y2 = [2469. 2313. 8026. 8738.  988. 7910. 2423.];

test_mAP_1 = [0.14113571 0.43854298 0.38669296 0.28549826 0.19594084 0.25010513 ...
0.34120721 0.43918955 0.37586004 0.05030128 0.12838038 0.09826902 ...
0.2766613  0.53998141 0.51714497 0.05626555 0.67150684 0.29225013 ...
0.2340637  0.18794656 0.28008585 0.58541808 0.2392784  0.26067112 ...
0.12227864 0.75973683 0.25660943 0.13610802 0.14178928 0.22127545 ...
0.04867352 0.14778741 0.41773421 0.23312476 0.14022737 0.27659439 ...
0.61173123 0.41907642 0.10199348 0.34961257 0.27580041 0.12486441];
test_AUC_1 = [0.68088341 0.90084224 0.92685426 0.94309756 0.86277537 0.887361 ...
0.84529155 0.95435917 0.90367205 0.8426957  0.83943473 0.85080758 ...
0.92736765 0.96803386 0.96818134 0.86878979 0.80649964 0.88761886 ...
0.85801472 0.83201315 0.880592   0.93609635 0.92657383 0.93269114 ...
0.95610643 0.93626166 0.88088971 0.73432005 0.85979934 0.88918731 ...
0.91911019 0.84672269 0.96231435 0.88545386 0.97579041 0.94310068 ...
0.92827017 0.93229892 0.87329362 0.85176906 0.93126981 0.88112978];
% AP_2 = [0.13510627 0.08951927 0.38546106 0.32151154 0.03130066 0.5521819 0.11340315];
% AUC_2 = [0.57488461 0.44802091 0.56031315 0.41099622 0.2862794  0.70135511 0.47959893];

test_mAP_1_val = sum(test_mAP_1 .* Y1 / sum(Y1))
% test_mAP_2 = test_mAP_2 .* Y2 / sum(Y2)

% load('siamese_doubleweighted_lambda1_1.5_lambda2_1_lambda3_0.25.mat')

[test_mAP_1, I1] = sort(test_mAP_1);
class_1_sorted = cell(1, 42);
for i = 1:42
    class_1_sorted{i} = class_1{I1(i)};
end

figure()
X1 = categorical(class_1_sorted);
X1 = reordercats(X1, class_1_sorted);

yyaxis left
plot(X1, test_mAP_1, 'kx', 'linewidth', 2.5, 'markersize', 12); hold off
ylim([0 1])
ylabel('Average Precision')
set(gca, 'ycolor', 'k', 'FontSize', 12)

yyaxis right
b1 = bar(X1, Y1(I1), 'facecolor', '#0072BD', 'edgecolor', 'none'); hold on
b1.FaceAlpha=0.25;
set(gca,'YScale','log')
ylim([20, 10^4])
ylabel('Number of audio clips')
set(gca, 'ycolor', '#0072BD')
grid on, grid minor


% [test_mAP_2, I2] = sort(test_mAP_2);
% class_2_sorted = cell(1, 7);
% for i = 1:7
%     class_2_sorted{i} = class_1{I2(i)};
% end

% figure()
% X2 = categorical(class_2_sorted);
% X2 = reordercats(X2, class_2_sorted);

% yyaxis left
% plot(X2, test_mAP_2, 'kx', 'linewidth', 2.5, 'markersize', 12); hold off
% ylabel('Average Precision')
% ylim([0 1])
% set(gca, 'ycolor', 'k', 'FontSize', 12)

% yyaxis right
% b1 = bar(X2, Y2(I2), 'facecolor', '#0072BD', 'edgecolor', 'none'); hold on
% b1.FaceAlpha=0.25;
% set(gca,'YScale','log')
% ylim([20, 10^4])
% ylabel('Number of audio clips')
% set(gca, 'ycolor', '#0072BD')
% grid on, grid minor

figure()
% X1 = categorical(class_1);

yyaxis left
plot(X1, test_AUC_1(I1), 'kx', 'linewidth', 2.5, 'markersize', 12); hold off
ylabel('AUC')
ylim([0 1])
set(gca, 'ycolor', 'k', 'FontSize', 12)

yyaxis right
b1 = bar(X1, Y1(I1), 'facecolor', '#0072BD', 'edgecolor', 'none'); hold on
b1.FaceAlpha=0.25;
set(gca,'YScale','log')
ylim([20, 10^4])
ylabel('Number of audio clips')
set(gca, 'ycolor', '#0072BD')
grid on, grid minor


% figure()
% X2 = categorical(class_2);

% yyaxis left
% plot(X2, test_AUC_2(I2), 'kx', 'linewidth', 2.5, 'markersize', 12); hold off
% ylabel('AUC')
% ylim([0 1])
% set(gca, 'ycolor', 'k', 'FontSize', 12)

% yyaxis right
% b1 = bar(X2, Y2(I2), 'facecolor', '#0072BD', 'edgecolor', 'none'); hold on
% b1.FaceAlpha=0.25;
% set(gca,'YScale','log')
% ylim([20, 10^4])
% ylabel('Number of audio clips')
% set(gca, 'ycolor', '#0072BD')
% grid on, grid minor


