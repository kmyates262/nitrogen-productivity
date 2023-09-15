%% Sensitivity Analysis
uqlab
close all
clearvars

load('lettuce_exp_matrices.mat')

BorgonovoOpts.Type = 'Sensitivity';
BorgonovoOpts.Method = 'Borgonovo';
BorgonovoOpts.Borgonovo.Method = 'HistBased';

BorgonovoOpts.Borgonovo.Sample.X = matrix1;
BorgonovoOpts.Borgonovo.Sample.Y = matrix2;
BorgonovoAnalysis= uq_createAnalysis(BorgonovoOpts);
sensitivities = BorgonovoAnalysis.Results.Delta; %Delta are the indices

save('lettuce_exp_sensitivity_results.mat','sensitivities')
bar(sensitivities)