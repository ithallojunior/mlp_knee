function n = normalize(e, min, max)

	n = (e .- (min)) ./ (max .- (min));

end


%!test
%!	assert(normalize([pi, pi/2, 0]) <= 1);
%!test
%!	assert(normalize([pi, pi/2, 0]) >= 0);
