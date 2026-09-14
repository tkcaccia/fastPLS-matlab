function value = backendInfo()
%BACKENDINFO Return the linear-algebra library used by this build.
value = string(fastpls_mex('backend'));
end
