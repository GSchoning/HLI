

def parse_des_file(iname):

    import numpy as np
    import re
    import pandas as pd


    info = {}

    with open(iname, 'r') as f:

        col_name_list            = []
        validgate_list_hm        = []
        validgate_list_lm        = []
        loop_geom_tx_list        = []
        transmitter_specs_dic    = {}
        receiver_specs_dic       = {}
        instrument_position_list = []
        tx_waveform_specs_list   = []
        txwaveform_lm_list       = []
        txwaveform_hm_list       = []
        
        flag_read_anything            = True
        flag_read_instrument_position = False
        flag_read_loop_geometry       = False
        flag_read_transmitterspecs    = False
        flag_read_tx_waveformspecs    = False
        flag_read_receiverspecs       = False
        flag_read_txwaveform_lm       = False
        flag_read_txwaveform_hm       = False
        flag_read_validgate_list_lm   = False
        flag_read_validgate_list_hm   = False
        flag_read_colnames            = False

        # read file line by line 
        while True:

            line = f.readline()

            # if it runs out of lines, stop reading the file
            if not line:
                break

            line = line.strip().lstrip('COMM').strip()

            if flag_read_anything:
                
                line = line.upper()

                if re.split(r'\s+', line)[0:3] == ['INSTRUMENT', 'POSITION', 'RELATIVE']:

                    flag_read_anything = False
                    flag_read_instrument_position = True

                    # discard header lines
                    while True:
                        line = f.readline()
                        if ")" in line:
                            break
                    continue
                
                elif re.split(r'\s+', line)[0:3] == ['TX', 'LOOP', 'GEOMETRY']:

                    flag_read_anything = False
                    flag_read_loop_geometry = True

                    # discard header lines
                    line = f.readline()
                    continue

                elif re.split(r"\s+", line) == ["TRANSMITTER", "SPECIFICATIONS"]:

                    flag_read_anything = False
                    flag_read_transmitterspecs = True

                    # no header lines to discard

                    continue

                elif re.split(r"\s+", line) == ["TX", "WAVEFORM", "SPECIFICATIONS"]:
                    flag_read_anything = False
                    flag_read_tx_waveformspecs = True

                    # no header lines to discard

                    continue

                elif re.split(r"\s+", line) == ["RECEIVER", "SPECIFICATIONS"]:
                    flag_read_anything = False
                    flag_read_receiverspecs = True

                    # no header lines to discard

                    continue

                elif re.split(r'\s+', line)[0:3] == ['LM', 'TX', 'WAVEFORM']:

                    flag_read_anything = False
                    flag_read_txwaveform_lm = True

                    # discard header lines
                    while True:
                        line = f.readline()
                        if "]" in line:
                            break
                    continue

                elif re.split(r"\s+", line)[0:3] == ['HM', 'TX', 'WAVEFORM']:
                    flag_read_anything = False
                    flag_read_txwaveform_hm = True

                    # discard header lines
                    while True:
                        line = f.readline()
                        if "]" in line:
                            break
                    continue
                
                elif re.split(r'\s+', line)[0:4] == ['LM', 'GATE', 'TIMES', 'SUPPLIED']:

                    flag_read_anything = False
                    flag_read_validgate_list_lm = True

                    # discard header lines
                    while True:
                        line = f.readline()
                        if ')' in line:
                            break
                    continue

                elif re.split(r'\s+', line)[0:4] == ['HM', 'GATE', 'TIMES', 'SUPPLIED']:

                    flag_read_anything = False
                    flag_read_validgate_list_hm = True

                    # discard header lines
                    while True:
                        line = f.readline()
                        if ')' in line:
                            break
                    continue
                
                elif re.split(r'\s+', line)[0:3] == ['FIELD', 'CHANNEL', 'DESCRIPTION']:

                    flag_read_anything = False
                    flag_read_colnames = True

                    continue

                # add here additional conditions to start flags to read particular blocks of data


            
            elif flag_read_instrument_position:

                if len(line) == 0:
                    # if the reader ran out of lines,
                    # stop reading and save the variables

                    flag_read_anything = True
                    flag_read_instrument_position = False

                    df_instrument_position = pd.DataFrame(instrument_position_list,
                                                          columns=['param', 'x_m', 'y_m', 'z_m'] )

                    df_instrument_position.iloc[:, 1:] = df_instrument_position.iloc[:, 1:].astype(float)

                    df_instrument_position.index = df_instrument_position['param']

                    info['df_instrument_position'] = df_instrument_position

                    continue
                
                line_items = re.split(r"\s+", line)

                line_items = [' '.join(line_items[0:-3])] + line_items[-3:]

                instrument_position_list.append(line_items)


            
            elif flag_read_loop_geometry:
                
                if len(line) == 0:
                    # if the reader ran out of lines,
                    # stop reading and save the variables

                    flag_read_anything = True
                    flag_read_loop_geometry = False

                    df_loop_geom_tx = pd.DataFrame(loop_geom_tx_list, columns=['x_m','y_m'])

                    df_loop_geom_tx = df_loop_geom_tx[:].astype(float)

                    info['df_loop_geom_tx'] = df_loop_geom_tx

                    continue

                line_items = re.split(r"\s+", line)

                loop_geom_tx_list.append(line_items)



            elif flag_read_transmitterspecs:
                
                if len(line) == 0:

                    # if the reader ran out of lines,
                    # stop reading and save the variables

                    flag_read_anything = True
                    flag_read_transmitterspecs = False

                    info['transmitter_specs'] = transmitter_specs_dic

                    continue
                
                line_items = re.split(r"\s{2,}", line)  # split where it finds two or more spaces
                transmitter_specs_dic[line_items[0]] = line_items[1]


            
            elif flag_read_tx_waveformspecs:

                if len(line) == 0 or "---" in line:

                    # if the reader ran out of lines,
                    # stop reading and save the variables

                    flag_read_anything = True
                    flag_read_tx_waveformspecs = False

                    df_tx_waveform_specs = pd.DataFrame(tx_waveform_specs_list, columns=['feature', 'lm', 'hm'])
                    df_tx_waveform_specs = df_tx_waveform_specs.set_index('feature')
                    info["tx_waveform_specs"] = df_tx_waveform_specs

                    continue
            
                line_items = re.split(r"\s{2,}", line)  # split where it finds two or more spaces
                if len(line_items) == 2:
                    split_position = line_items[1].find('HM')
                    line_items = [line_items[0], line_items[1][:split_position], line_items[1][split_position:]]
                    line_items = [x.strip() for x in line_items]

                line_items[0] = line_items[0].lower().strip()
                line_items[1] = line_items[1].lower().removeprefix("lm =").strip()
                line_items[2] = line_items[2].lower().removeprefix("hm =").strip()

                tx_waveform_specs_list.append(line_items)
            
            
            
            elif flag_read_receiverspecs:

                if len(line) == 0 or "---" in line:
                    # if the reader ran out of lines,
                    # stop reading and save the variables

                    flag_read_anything = True
                    flag_read_receiverspecs = False

                    info["receiver_specs"] = receiver_specs_dic

                    continue

                line_items = re.split(r"\s{2,}", line)

                receiver_specs_dic[line_items[0]] = line_items[1]

            
            elif flag_read_txwaveform_lm:

                if len(line) == 0 or '---' in line:

                    # if the reader ran out of lines,
                    # stop reading and save the variables

                    flag_read_anything = True
                    flag_read_txwaveform_lm = False

                    df_txwaveform_lm = pd.DataFrame(txwaveform_lm_list,
                                                    columns=["time_s",
                                                             "amplitude"] )
                    
                    info["df_txwaveform_lm"] = df_txwaveform_lm

                    continue

                line_items = re.split(r"\s+", line)

                line_items = [float(x) for x in line_items]

                txwaveform_lm_list.append(line_items)
            


            elif flag_read_txwaveform_hm:

                if len(line) == 0:

                    # if the reader ran out of lines,
                    # stop reading and save the variables

                    flag_read_anything = True
                    flag_read_txwaveform_hm = False

                    df_txwaveform_hm = pd.DataFrame(txwaveform_hm_list, 
                                                    columns=["time_s", "amplitude"] )

                    info['df_txwaveform_hm'] = df_txwaveform_hm

                    continue

                line_items = re.split(r'\s+', line)

                line_items = [float(x) for x in line_items]

                txwaveform_hm_list.append(line_items)




            elif flag_read_validgate_list_lm:

                if len(line) == 0:

                    # if the reader ran out of lines, 
                    # stop reading and save the variables

                    flag_read_anything = True
                    flag_read_validgate_list_lm = False

                    df_validgates_lm = pd.DataFrame(validgate_list_lm,
                                                    columns=["gate",
                                                             "width_us",
                                                             "open_us",
                                                             "centre_us",
                                                             "close_us"])

                    df_validgates_lm["gate"] = df_validgates_lm["gate"].astype(int)

                    df_validgates_lm.index = df_validgates_lm["gate"]

                    info["df_validgates_lm"] = df_validgates_lm

                    validgate_list_lm = df_validgates_lm["gate"].to_list()

                    info["validgate_list_lm"] = validgate_list_lm

                    continue
                
                line_items = re.split(r"\s+", line)

                line_items = [float(x) for x in line_items]
                validgate_list_lm.append(line_items)

                pass



            elif flag_read_validgate_list_hm:

                if len(line) == 0:

                    # if the reader ran out of lines,
                    # stop reading and save the variables

                    flag_read_anything = True
                    flag_read_validgate_list_hm = False

                    df_validgates_hm = pd.DataFrame(validgate_list_hm, 
                                                    columns=['gate',
                                                             'width_us',
                                                             'open_us',
                                                             'centre_us',
                                                             'close_us'])
                    
                    df_validgates_hm["gate"] = df_validgates_hm["gate"].astype(int)

                    df_validgates_hm.index = df_validgates_hm["gate"]

                    info["df_validgates_hm"] = df_validgates_hm

                    validgate_list_hm = df_validgates_hm["gate"].to_list()

                    info["validgate_list_hm"] = validgate_list_hm

                    continue
                
                line_items = re.split(r"\s+", line)

                line_items = [float(x) for x in line_items]
                validgate_list_hm.append(line_items)

                pass

            elif flag_read_colnames:

                if len(line) == 0:

                    # If the reader ran out of lines, 
                    # stop reading.
                    # These lines may never execute though, since it's the end of the file.
                    
                    flag_read_anything = True
                    flag_read_colnames = False
                    continue
                
                line_items = re.split(r"\s+", line)

                # make list of the column names

                if ':' not in line_items[0]:

                    col_name_list.append(line_items[1])

                elif ':' in line_items[0]:

                    col_start, col_end = [ int(x) for x in re.split(r':', line_items[0]) ]

                    ncols = col_end - col_start + 1

                    for ncol in range(ncols):

                        col_name_list.append(f'{line_items[1]}_{ncol + 1}')

                # read the number of gates

                if line_items[1].upper() == 'LM_Z':

                    info['n_lm_gates'] = int( line_items[13].upper().split('F')[0] )
                
                elif line_items[1].upper() == "HM_Z":
                    info["n_hm_gates"] = int(line_items[13].upper().split("F")[0])

        info['col_name_list'] = col_name_list
