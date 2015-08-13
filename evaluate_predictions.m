
function evaluate_predictions(target_file, prediction_file, report_file, description)

    addpath(genpath('matlab/'));
    addpath(genpath('matlab/seunglab/'));
    addpath(genpath('matlab/seunglab/segmentation/'));

    initial_thresholds = 0.1:0.1:1.0;
    min_step = 0.001;
%    min_step = 0.1;
    
    f = fopen([report_file '/errors_new.txt'], 'w');
    saveAndPrint(f, 'Description:\n%s\n\n', description);
    
    fprintf('calculating rand Index\n')
    [r_thresholds, r_fscore, r_tp, r_fp, r_pos, r_neg, ~] = ...
        evaluate_thresholds(target_file, prediction_file, initial_thresholds, 'rand', min_step);
    
    fprintf('calculating pixel error\n')
    [p_thresholds, p_err, p_tp, p_fp, p_pos, p_neg, p_sqerr] = ...
        evaluate_thresholds(target_file, prediction_file, initial_thresholds, 'pixel', min_step);
    
    save([report_file '/errors_new.mat'], ...
        'r_thresholds', 'r_fscore', 'r_tp', 'r_fp', 'r_pos', 'r_neg', ...
        'p_thresholds', 'p_err', 'p_tp', 'p_fp', 'p_pos', 'p_neg', 'p_sqerr');
    
    saveAndPrint(f, 'Mean Pixel Square Error: %f\n', p_sqerr);
    
    [e, idx] = max(r_fscore);
    best_threshold = r_thresholds(idx);
    saveAndPrint(f, 'Best Threshold for Rand F-score: %f\n', best_threshold);
    saveAndPrint(f, 'Best Rand F-score: %f\n', e);
    
    [e, idx] = min(p_err);
    best_threshold = p_thresholds(idx);
    saveAndPrint(f, 'Best Threshold for Pixel Error: %f\n', best_threshold);
    saveAndPrint(f, 'Best Pixel Error: %f\n', e);
    
    min_threshold_idx = floor((length(initial_thresholds)+1)/2);
    max_threshold_idx = ceil(length(r_thresholds) + 1 - min_threshold_idx);
        
    clf;
    close all
    plt = subplot(2,2,1);
    plot(r_thresholds, r_fscore);
    title('Rand F-Score');
    xlabel('Threshold');
    ylabel('Rand F-Score');
    xlim([r_thresholds(min_threshold_idx), r_thresholds(max_threshold_idx)]);
    ylim([0, 1]);
    
    subplot(2,2,2);
    r_fp_rate = r_fp/r_neg;
    [r_fp_rate, idxs] = sort(r_fp_rate);
    r_tp_rate = r_tp(idxs)/r_pos;
    while(any(diff(r_tp_rate)<=0))
        r_fp_rate = r_fp_rate(diff([-inf r_tp_rate])>0);
        r_tp_rate = r_tp_rate(diff([-inf r_tp_rate])>0);
    end
    plot(r_fp_rate, r_tp_rate);
    title('Rand Error ROC');
    xlabel('False Positive');
    ylabel('True Positive');
    xlim([0, 1]);
    ylim([0, 1]);
    
    subplot(2,2,3);
    plot(p_thresholds, p_err);
    title('Pixel Error');
    xlabel('Threshold');
    ylabel('Pixel Error');
    xlim([p_thresholds(min_threshold_idx), p_thresholds(max_threshold_idx)]);
    ylim([0, 1]);

    subplot(2,2,4);
    plot(p_fp/p_neg, p_tp/p_pos);
    title('Pixel Error ROC');
    xlabel('False Positive');
    ylabel('True Positive');
    xlim([0, 1]);
    ylim([0, 1]);
    
    saveas(plt, [report_file '/errors_new.fig']);
    saveas(plt, [report_file '/errors_new.png'], 'png');
    
    fclose(f);
    
    
end

function saveAndPrint(varargin)
    file = varargin{1};
    fprintf(varargin{2:end});
    fprintf(file, varargin{2:end});
end

function [thresholds, err, tp, fp, pos, neg, p_sqerr] = ...
evaluate_thresholds(target_file, prediction_file, thresholds, randOrPixel, min_step)
    M = 1;
    err = nan(M, length(thresholds));
    tp = nan(M, length(thresholds));
    fp = nan(M, length(thresholds));
    pos = nan(M, 1);
    neg = nan(M, 1);
    p_sqerr = nan(M, 1);
    n_examples = nan(M, 1);
    %[err, tp, fp, pos, neg, p_sqerr] = get_stats(files{1}, thresholds, dims, randOrPixel);
    parfor i=1:M
        %for i=1:length(files)
        [err_, tp_, fp_, pos_, neg_, p_sqerr_, n_examples_] = get_stats(target_file, prediction_file, thresholds, randOrPixel);
        err(i,:) = err_;
        tp(i,:) = tp_;
        fp(i,:) = fp_;
        pos(i) = pos_;
        neg(i)= neg_;
        p_sqerr(i) = p_sqerr_;
        n_examples(i) = n_examples_;
        %fprintf(':O');
    end
    
%      tp = sum(bsxfun(@times, tp, n_examples));
%     fp = sum(bsxfun(@times, fp, n_examples));
    tp=tp*n_examples;
    fp=fp*n_examples;
    pos = sum(pos .* n_examples);
    neg = sum(neg .* n_examples);
    p_sqerr = sum(p_sqerr .* n_examples) / sum(n_examples);
    if(strcmp(randOrPixel,'rand'))
        prec = tp ./ (tp + fp);
        rec = tp / pos;
        err = 2 * (prec .* rec) ./ (prec + rec);
        [best_err, idx] = max(err);
    else
%        err = sum(bsxfun(@times, err, n_examples)) ./ sum(n_examples);
        err = (err*n_examples) ./ sum(n_examples);
        [best_err, idx] = min(err);
    end

    step = thresholds(2) - thresholds(1);
    best_threshold = thresholds(idx);
    fprintf('Thresholds = %f:%f:%f, Best %s = %f\n', thresholds(1), step, thresholds(end), randOrPixel, best_err);

    if(step > min_step)
        new_step = 2 * step/(length(thresholds)-1);
        inner_thresholds = best_threshold-step:new_step:best_threshold+step;
        [thresholds_, err_, tp_, fp_, ~, ~, ~] = ...
            evaluate_thresholds(target_file, prediction_file, inner_thresholds, randOrPixel, min_step);
        
        thresholds = [thresholds(1:idx-1), thresholds_, thresholds(idx+1:end)];
        err = [err(1:idx-1), err_, err(idx+1:end)];
        tp = [tp(1:idx-1), tp_, tp(idx+1:end)];
        fp = [fp(1:idx-1), fp_, fp(idx+1:end)];
    else
        fprintf('\n');
    end
end


function [err, tp, fp, pos, neg, p_sqerr, n_examples] = get_stats(target_file, prediction_file, thresholds, randOrPixel)
    
    [ affTrue, affEst, dimensions ] = load_affs( target_file, prediction_file);
    n_examples = numel(affEst)/2;
%     load([file '/gradients.txt'])
%     [~, idxs_idxs] = sort(sum(gradients(:,2:4), 2));
%     idxs = gradients(idxs_idxs, 1);
%     foo = sum(predictions,2)/3;
%     foo(idxs(1)) = 3;
%     bar = permute(reshape(foo, fliplr(dimensions)), [3,2,1]);
%     %BrowseComponents('ii', affEst, bar);
%     
%     load('/home/luke/Documents/asdf1/seg.txt')
%     df = int32(seg);
%     load('/home/luke/Documents/asdf2/seg.txt')
%     dt = int32(seg);
%     asdf = zeros(size(labels, 1), 1);
%     asdf(df(:, 1) + 1) = 1;
%     asdf(dt(:, 1) + 1) = 2;
%     asdf = permute(reshape(asdf, fliplr(dimensions)), [3,2,1]);
%     
%     load([file '/seg.txt'])
%     seg = int32(seg);
%     segs = zeros(size(seg, 1), 1);
%     segs(seg(:, 1) + 1) = seg(:, 2);
%     segs = permute(reshape(segs, fliplr(dimensions)), [3,2,1]);
%     BrowseComponents('ic', bar, asdf)
    
    if(strcmp(randOrPixel,'rand'))
        nHood = eye(3);
        compTrue = flip_aff(connectedComponents(flip_aff(affTrue), nHood));  
%        compTrue = flip_aff(bwconncomp(flip_aff(affTrue)));
        p_sqerr = -1;
    else
        p_sqerr = (affEst(:) - affTrue(:))' * (affEst(:) - affTrue(:)) / numel(affEst);
    end
    
    err = zeros(size(thresholds));
    tp = zeros(size(thresholds));
    fp = zeros(size(thresholds));
    pos = zeros(size(thresholds));
    neg = zeros(size(thresholds));
    
    threshWorkers = length(thresholds);
    if(strcmp(randOrPixel,'rand'))   
        parfor i=1:threshWorkers
            threshold = thresholds(i);
                [err_, tp_, fp_, pos_, neg_] = ...
                    get_rand_stats_for_threshold(compTrue, affEst, threshold);
                err(i) = err_;
                tp(i) = tp_;
                fp(i) = fp_;
                if(~isnan(pos_)) pos(i)=pos_; end
                if(~isnan(neg_)) neg(i)=neg_; end
        end
    else
        parfor i=1:threshWorkers
            threshold = thresholds(i);
            [err_, tp_, fp_, pos(i), neg(i)] = ...
                get_pixel_stats_for_threshold(affTrue, affEst, threshold);
            err(i) = err_;
            tp(i) = tp_;
            fp(i) = fp_;
        end
    end
    pos = nanmax(pos);
    neg = nanmax(neg);
end

function a = flip_aff(M)
    a = M(end:-1:1, end:-1:1, end:-1:1, :);
end

function [p_err, p_tp, p_fp, p_pos, p_neg] = ...
        get_pixel_stats_for_threshold(affTrue, affEst, threshold)
    p_err = 1-sum((affEst(:)>threshold)==affTrue(:))/numel(affTrue);
    p_tp = sum((affEst(:)>threshold) & affTrue(:));
    p_fp = sum((affEst(:)>threshold) & ~affTrue(:));

    p_pos = sum(affTrue(:));
    p_neg = sum(~affTrue(:));
end

function [r_err, r_tp, r_fp, r_pos, r_neg] = ...
        get_rand_stats_for_threshold(compTrue, affEst, threshold)

    if(~ all(affEst(:)<=threshold) && ~ all(affEst(:)>threshold))
        nHood = -eye(3);
        sz = size(affEst);
        affEst = reshape(affEst,[sz(1) sz(2) sz(3) sz(end)]);
 
        compEst = flip_aff(connectedComponents(flip_aff(affEst)>threshold, nHood));
%        compEst = flip_aff(bwconncomp(flip_aff(affEst)>threshold, nHood));
        watershed = markerWatershed(affEst, nHood, compEst);

        [ri, stats] = randIndex(compTrue, watershed);
        r_err = 1-ri;
        r_tp = stats.truePos;
        r_fp = stats.falsePos;

        r_pos = stats.pos;
        r_neg = stats.neg;
%         r_fscore = 2 * (stats.prec * stats.rec) / (stats.prec + stats.rec);
    else 
        r_err = NaN;
        r_tp = NaN;
        r_fp = NaN;

        r_pos = NaN;
        r_neg = NaN;
%         r_fscore = -1;
    end
end
