%% Sensitivity Analysis
uqlab
close all
clearvars

mat_files = dir(fullfile('.','*matrices.mat'));
crop_names = ["dry_bean", "lettuce", "peanut", "rice", "soybean", "sweet_potato", "tomato", "wheat", "white_potato"];

for k = 1:length(mat_files)
    fullFileName = fullfile('.', mat_files(k).name);
    load(fullFileName)
    
    BorgonovoOpts.Type = 'Sensitivity';
    BorgonovoOpts.Method = 'Borgonovo';
    BorgonovoOpts.Borgonovo.Method = 'HistBased';
    
    BorgonovoOpts.Borgonovo.Sample.X = matrix1;
    BorgonovoOpts.Borgonovo.Sample.Y = matrix2;
    BorgonovoAnalysis= uq_createAnalysis(BorgonovoOpts);
    sensitivities = BorgonovoAnalysis.Results.Delta;
    
    file_out = strcat(crop_names(k),"_sensitivity_results.mat");
    save(file_out,'sensitivities');
end
