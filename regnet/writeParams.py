import os





def writeSummary(data_dir, data_description, data_purpose,

                 targetSelection,

                 num_c1, c1_low, c1_high,

                 num_T21, T21_low, T21_high,

                 num_T22, T22_low, T22_high,

                 num_times, time_low, time_high,

                 SNR, signal_type,

                 num_reg_params, delta, start_exp_shift, end_exp_shift,

                 frac_trivial_tol, frac_error_tol,

                 weights, init_nlls_guess,

                 num_triplets, num_noise_realizations,

                 notes):

    """

    WRITEPARAMS writes a text file that gives a summary of the parameters

    used when making a data set.

    """



    # Name of the file to write

    param_file = data_description+data_purpose+'.summary.txt'



    # If the file exists, just append the purpose, num triples, and num...

    # ...noise realizations

    if not os.path.exists(data_dir+param_file):

        # Open file to append text to

        f = open(data_dir+param_file, 'a')

        f.write('Parameter Space\n')

        f.write('---------------\n')

        f.write(f'(num_c1, c1_low, c1_high) = ({num_c1}, {c1_low}, {c1_high})\n')

        f.write(f'(num_T21, T21_low, T21_high) = ({num_T21}, {T21_low}, {T21_high})\n')

        f.write(f'(num_T22, T22_low, T22_high) = ({num_T22}, {T22_low}, {T22_high})')        

        f.write('\n\n')

        f.write('Times, SNR, Signal Type\n')

        f.write('-----------------------\n')

        f.write(f'(num_times, time_low, time_high) =  ({num_times}, {time_low}, {time_high})\n')

        f.write(f'SNR = {SNR}\n')

        f.write(f'mySignalType = {signal_type}')

        f.write('\n\n')

        f.write('NLLS params\n')

        f.write('-----------\n')

        f.write('-- Regularization parameters\n')

        f.write(f'traj_len = {num_reg_params}\n')

        f.write(f'delta = {delta}\n')

        f.write(f'(start_exp_shift, end_exp_shift) = ({start_exp_shift}, {end_exp_shift})\n')

        f.write('-- Weights in penalty term\n')

        f.write(f'D = {weights}\n')

        f.write(f'initial NLLS guess = {init_nlls_guess}')

        f.write('\n\n')

        f.write(f'frac_trivial_tol = {frac_trivial_tol}\n')

        f.write(f'frac_error_tol = {frac_error_tol}')

        f.write('\n\n')





    else:

        # Open file to append text

        f = open(data_dir+param_file, 'a')



    f.write('(purpose, targetSelection, num_triplets, num_noise_realizations) = '+

            f'({data_purpose}, {targetSelection}, {num_triplets}, {num_noise_realizations})\n')

    if notes: f.write(f'    -> Notes: {notes}\n\n')

    f.close()

