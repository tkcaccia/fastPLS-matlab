function values = vip(model)
%VIP Return variable-importance paths from a fitted fastPLS model.
arguments
    model (1, 1) fastpls.Model
end
values = model.vip();
end
