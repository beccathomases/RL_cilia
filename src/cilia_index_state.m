function c = cilia_index_state(s, cmin, cmax)
%CILIA_INDEX_STATE Inverse of cilia_state_index.

    vals = cmin:cmax;
    nVals = numel(vals);

    s0 = s - 1;
    i0 = mod(s0, nVals);
    i1 = mod(floor(s0/nVals), nVals);
    i2 = floor(s0/(nVals^2));

    c = [vals(i0+1); vals(i1+1); vals(i2+1)];
end
