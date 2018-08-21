function yyxx = initBilinearGrids(inputSize)
    yi = linspace(-1, 1, inputSize);
    xi = linspace(-1, 1, inputSize);
    [xx, yy] = meshgrid(xi, yi);
    yyxx = single([yy(:), xx(:)]');
end

