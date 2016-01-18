function res = smoothen(labels, Sp)
segs = unique(Sp);
res = zeros(size(labels));
for i=segs'
  idx = find(Sp == i);
  res(idx) = mode(labels(idx));
end