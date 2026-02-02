function rfft = computeFFT(input, N)
input_w_full = fft(input)/N;

input_w = input_w_full(1:N/2+1,:);
input_w(2:end-1,:) = 2*input_w(2:end-1,:);

rfft = input_w;
end