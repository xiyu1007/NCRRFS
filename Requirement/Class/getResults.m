function results = getResults(n)
    results.acc = NaN(1,n);
    results.sen = NaN(1,n);
    results.spe = NaN(1,n);
    results.f1 = NaN(1,n);
    results.auc = NaN(1,n);
    results.labs = cell(1,n);
    results.decs = cell(1,n);
end
