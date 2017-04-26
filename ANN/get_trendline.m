function [ xPts,yPts ] = get_trendline( accuracies,start_pt,end_pt,deg )
    coeffs = polyfit(accuracies(:,1), accuracies(:,2),deg);
    xPts = start_pt:0.01:end_pt;
    yPts = polyval(coeffs, xPts);
end

