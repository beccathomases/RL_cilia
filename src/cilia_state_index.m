function s = cilia_state_index(c, cmin, cmax)
%CILIA_STATE_INDEX Map c=[c0;c1;c2] in [cmin,cmax]^3 to 1..S

    vals = cmin:cmax;
    nVals = numel(vals);

    % convert coefficient value to 0-based index
    i0 = c(1) - cmin;  % 0..nVals-1
    i1 = c(2) - cmin;
    i2 = c(3) - cmin;

    if any([i0,i1,i2] < 0) || any([i0,i1,i2] > (nVals-1))
        error('Coefficient out of bounds.');
    end

    % base-n encoding: s = 1 + i0 + n*i1 + n^2*i2
    s = 1 + i0 + nVals*i1 + (nVals^2)*i2;
end
