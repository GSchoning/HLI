# fmt: off
import numpy as np
import pandas as pd


def calculate_vertical_correction_numpy(tilt_angle_degrees, distance_to_receiver):
    """
    Calculates the vertical correction using NumPy. Handles arrays.
    Clips input angles to [-90, 90] and handles 90-degree cases.

    A POSITIVE tilt angle means the receiver is LOWER (as per YOUR convention).

    Args:
        tilt_angle_degrees: NumPy array or scalar, tilt angle(s) in degrees.
        distance_to_receiver: NumPy array or scalar, distance(s) to receiver.

    Returns:
        NumPy array or scalar, vertical correction(s).
        Returns None if inputs are not compatible.
    """

    # Input Validation: Check for NumPy arrays or scalar values, and consistent shapes if both are arrays
    if isinstance(tilt_angle_degrees, np.ndarray) and isinstance(distance_to_receiver, np.ndarray):
        if tilt_angle_degrees.shape != distance_to_receiver.shape:
            return None # Inconsistent shapes
    elif not isinstance(tilt_angle_degrees, (np.ndarray, (int, float))) or not isinstance(distance_to_receiver, (np.ndarray, (int, float))):
        return None # Invalid input types

    # Clip tilt angles to [-90, 90] to avoid issues with tan()
    tilt_angle_degrees = np.clip(tilt_angle_degrees, -90, 90)  # Handles input ranges outside [-90, 90]

    tilt_angle_radians = np.radians(tilt_angle_degrees)

    # Handle 90-degree cases using np.where for array or scalar inputs
    # NO NEGATIVE SIGN here for YOUR convention:
    vertical_correction = np.where(np.abs(tilt_angle_degrees) == 90, None, distance_to_receiver * np.tan(tilt_angle_radians))

    return vertical_correction


if __name__ == "__main__":
    # import this way if this script is executed directly - for debugging
    from des_parser import parse_des_file
    from gex_parser import parse_gex_file
else:
    # import this way during normal use - when this script is imported by the main one
    from libraries.des_parser import parse_des_file
    from libraries.gex_parser import parse_gex_file



class Data:
    # default location for DFILE r"data\All_AVG_export.gz.parquet"
    dstart = 20 - 2
    lmcol = 85
    hmcol = 110
    runc_offset=0.03

    def __init__(self, sounding_averaging=None):
        self.sounding_averaging = sounding_averaging
        pass

    def proc_parquet(self, Survey, DFILE):
        """This dunction processs parquet file into AEM survey Class data

        Args:
           Survey (Survey Class): Survey CLass as defined by this module
           DFILE (file): location of parquet file
        """

        data_file = pd.read_parquet(DFILE)
        data_file.columns = [x.strip(" ") for x in data_file.columns]
        data_file[data_file == 9999] = np.nan
        data_file[data_file == "*"] = np.nan
        data_file[data_file == -9999] = np.nan
        data_file["TIME"] = data_file["TIME"].str.strip()

        station_data = (
            data_file[
                [
                    "/       DATE",
                    "TIME",
                    "LINE_NO",
                    "UTMX",
                    "UTMY",
                    "ELEVATION",
                    "RX_ALTITUDE",
                    "TX_ALTITUDE",
                    "RX_ALTITUDE_STD",
                    "TX_ALTITUDE_STD",
                ]
            ]
            .set_index(["LINE_NO", "TIME"])
            .drop_duplicates()
        )
        CH1 = data_file[data_file.CHANNEL_NO == 1].set_index(["LINE_NO", "TIME"])
        CH2 = data_file[data_file.CHANNEL_NO == 2].set_index(["LINE_NO", "TIME"])

        lmstart = self.dstart + Survey.skipgates - 1
        lmend = lmstart + Survey.n_lm_gates

        lmuncstart = self.lmcol + Survey.skipgates - 1 - 2
        lmuncend = lmuncstart + Survey.n_lm_gates

        hmstart = (
            self.dstart + Survey.skipgates + Survey.n_lm_gates + Survey.skipgates - 1
        )
        hmend = hmstart + Survey.n_hm_gates

        hmuncstart = self.hmcol + Survey.skipgates - 1 - 2
        hmuncend = hmuncstart + Survey.n_hm_gates

        lm_data = CH1.iloc[:, lmstart:lmend]
        hm_data = CH2.iloc[:, hmstart:hmend]

        lm_relunc = CH1.iloc[:, lmuncstart:lmuncend]
        hm_relunc = CH2.iloc[:, hmuncstart:hmuncend]

        self.station_data = station_data
        self.hm_data = hm_data
        self.lm_data = lm_data
        self.lm_relunc = lm_relunc
        self.hm_relunc = hm_relunc
        self.lm_std = self.lm_relunc * self.lm_data.values
        self.hm_std = self.hm_relunc * self.hm_data.values

    def proc_xyz(self, Survey, DFILE):
        """This function processes XYZ files into AEM survey Class data"""
        
        # 1. READ HEADERS: Count comment lines and find exact column names
        with open(DFILE, 'r') as f:
            skip_lines = 0
            header_line = None
            for line in f:
                if line.startswith('/'):
                    skip_lines += 1
                    if 'Easting' in line:
                        # Grab the column names
                        header_line = line.replace('/', '').strip().split()
                else:
                    # We hit the real data numbers, stop reading
                    break 
                
        # 2. LOAD DATA: Skip the comment block and handle '*' as NaN
        df = pd.read_csv(
            DFILE, 
            sep=r'\s+', 
            skiprows=skip_lines, 
            header=None, 
            names=header_line, 
            na_values=['*', '***', '9999', '-9999']
        )
        
        # 3. RENAME COLUMNS to match the internal standard
        mapping = {
            'Easting': 'UTMX',
            'Northing': 'UTMY',
            'DEM_AHD': 'ELEVATION',
            'Line': 'LINE_NO',
            'Time': 'TIME',
            'Date': 'DATE',
            'Height': 'HEIGHT_RAW' 
        }
        df = df.rename(columns=mapping)

        # 4. CALCULATE GEOMETRY (Vertical Correction)
        df["RX_ALTITUDE"] = df["HEIGHT_RAW"] + calculate_vertical_correction_numpy(
            df["AngleX"].values, Survey.rx_offset[0]
        )
        df["TX_ALTITUDE"] = df["HEIGHT_RAW"]
        df['RX_ALTITUDE_STD'] = 0.1
        df['TX_ALTITUDE_STD'] = 0.1

        # 5. SORT AND AVERAGE
        df = df.sort_values(["LINE_NO", "TIME"]).reset_index(drop=True)
        
        if self.sounding_averaging is None or self.sounding_averaging <= 1:
            self.sounding_averaging = None
            
        if self.sounding_averaging:
            def average_data_file(data_file, sounding_averaging):
                line_list = data_file['LINE_NO'].unique().tolist()
                data_file_averaged_list = []
                
                for line in line_list:
                    filt1 = data_file['LINE_NO'] == line
                    data_file_line = data_file[filt1].reset_index(drop=True)
                    
                    # Ensure DATE and TIME are not deleted by averaging
                    agg_dict = {col: 'mean' if pd.api.types.is_numeric_dtype(data_file_line[col]) else 'first' 
                                for col in data_file_line.columns}
                    
                    data_file_line_averaged = (data_file_line
                                .groupby(np.arange(len(data_file_line)) // sounding_averaging)
                                .agg(agg_dict) 
                                .reset_index(drop=True))
                    
                    data_file_averaged_list.append(data_file_line_averaged)

                if data_file_averaged_list:
                    data_file_averaged = pd.concat(data_file_averaged_list, axis=0, ignore_index=True)
                else:
                    data_file_averaged = pd.DataFrame()
                return data_file_averaged

            df = average_data_file(data_file=df, sounding_averaging=self.sounding_averaging)

        # 6. EXTRACT GATES 
        # Using n_lm_gates (calculated in proc_gex) to match the 0-indexed XYZ headers
        lm_cols_dataonly = [f"LM_Z_dBdt[{i}]" for i in range(Survey.skipgates,Survey.n_lm_gates+Survey.skipgates)]
        hm_cols_dataonly = [f"HM_Z_dBdt[{i}]" for i in range(Survey.skipgates,Survey.n_hm_gates+Survey.skipgates)]
        
        lm_cols_unconly = [f"RelUnc_LM_Z_dBdt[{i}]" for i in range(Survey.skipgates,Survey.n_lm_gates+Survey.skipgates)]
        hm_cols_unconly = [f"RelUnc_HM_Z_dBdt[{i}]" for i in range(Survey.skipgates,Survey.n_hm_gates+Survey.skipgates)]

        # 7. INDEX THE DATAFRAME
        df_indexed = df.set_index(["LINE_NO", "TIME"])

        # 8. UNIT CONVERSION (pico-volts to volts)
        df_indexed[lm_cols_dataonly] = df_indexed[lm_cols_dataonly] * 1e-12
        df_indexed[hm_cols_dataonly] = df_indexed[hm_cols_dataonly] * 1e-12

        # 9. ASSIGN CLASS VARIABLES
        self.station_data = df[["DATE", "TIME", "LINE_NO", "UTMX", "UTMY", 
                                "ELEVATION", "RX_ALTITUDE", "TX_ALTITUDE", 
                                "RX_ALTITUDE_STD", "TX_ALTITUDE_STD",
                                "AngleX", "AngleY"]].set_index(["LINE_NO", "TIME"]).drop_duplicates()
                                
        self.lm_data = df_indexed[lm_cols_dataonly]
        self.hm_data = df_indexed[hm_cols_dataonly]

        # Handle Uncertainties (fall back to offset if missing)
        if all(col in df_indexed.columns for col in lm_cols_unconly):
            self.lm_relunc = df_indexed[lm_cols_unconly] + self.runc_offset
            self.hm_relunc = df_indexed[hm_cols_unconly] + self.runc_offset
        else:
            self.lm_relunc = self.lm_data * 0 + self.runc_offset
            self.hm_relunc = self.hm_data * 0 + self.runc_offset

        self.lm_std = self.lm_relunc * self.lm_data.values
        self.hm_std = self.hm_relunc * self.hm_data.values

        # Store column names so lm_data_() works
        self.lm_cols_dataonly = lm_cols_dataonly
        self.hm_cols_dataonly = hm_cols_dataonly
        # 10. CLEAN UP DUMMY GATES (Remove all-NaN columns)
        # Find which columns actually contain data
        valid_lm_mask = ~self.lm_data.isna().all()
        valid_hm_mask = ~self.hm_data.isna().all()

        # Filter the DataFrames
        self.lm_data = self.lm_data.loc[:, valid_lm_mask]
        self.hm_data = self.hm_data.loc[:, valid_hm_mask]

        # Filter uncertainties using the positional boolean array (.values) 
        # because the column names differ ('LM_Z_dBdt' vs 'RelUnc_LM_Z_dBdt')
        self.lm_relunc = self.lm_relunc.iloc[:, valid_lm_mask.values]
        self.hm_relunc = self.hm_relunc.iloc[:, valid_hm_mask.values]

        # Recalculate Standard Deviations
        self.lm_std = self.lm_relunc * self.lm_data.values
        self.hm_std = self.hm_relunc * self.hm_data.values

        # Store the cleaned column lists so lm_data_() works
        self.lm_cols_dataonly = self.lm_data.columns.tolist()
        self.hm_cols_dataonly = self.hm_data.columns.tolist()

        # CRITICAL: Trim the Survey time arrays so the inversion physics stay aligned
        Survey.lm_times = Survey.lm_times[valid_lm_mask.values]
        Survey.hm_times = Survey.hm_times[valid_hm_mask.values]
        
        # Update gate counts for the rest of the script
        Survey.n_lm_gates = len(Survey.lm_times)
        Survey.n_hm_gates = len(Survey.hm_times)

        return None

        return None
    
    def proc_dat(self, Survey, DFILE, sounding_averaging=None):

        data_file = pd.read_csv(DFILE, sep="\s+", header=None, names=Survey.col_name_list)

        data_file = data_file.rename(columns={'Date_Local':'DATE',
                                              'Time_Local':'TIME',
                                              'Line':'LINE_NO',
                                              'Easting':'UTMX',
                                              'Northing':'UTMY',
                                              'DTM_AHD':'ELEVATION',
                                              '':'RX_ALTITUDE',
                                              '':'TX_ALTITUDE',
                                              '':'RX_ALTITUDE_STD',
                                              '':'TX_ALTITUDE_STD',
                                              'AngleX':'AngleX',
                                              'AngleY':'AngleY' } ) 
        
        # WARNING: The following fields were not present in the data table
        #          so I initialised these with arbitratry values. 
        #          These will need to be corrected.
        # RX_ALTITUDE
        # TX_ALTITUDE
        # RX_ALTITUDE_STD
        # TX_ALTITUDE_STD
        data_file["RX_ALTITUDE"] = data_file["Height"]+calculate_vertical_correction_numpy(data_file.AngleX.values,Survey.rx_offset[0])
        #data_file["GPS_Alt"] - data_file["ELEVATION"]    # WARNING: to be corrected
        data_file["TX_ALTITUDE"] = data_file["Height"]  # data_file["GPS_Alt"] - data_file["ELEVATION"]    # WARNING: to be corrected
        data_file['RX_ALTITUDE_STD'] = 0.1                 # WARNING: to be corrected
        data_file['TX_ALTITUDE_STD'] = 0.1                 # WARNING: to be corrected


        # SORT ROWS IN CASE THEY ARE NOT SORTED
        data_file = data_file.sort_values(["LINE_NO", "TIME"]).reset_index(drop=True)


        # AVERAGING MULTIPLE ROWS IF NEEDED

        # if averaging is 1 or less, set the value to 'None'
        if self.sounding_averaging is None or self.sounding_averaging <= 1:
            self.sounding_averaging = None
        
        # if averaging is 2 or more, apply it
        if self.sounding_averaging:

            def average_data_file(data_file, sounding_averaging):

                line_list = data_file['LINE_NO'].unique().tolist()
                
                data_file_averaged_list = []
                
                # process the lines one by one
                for line in line_list:
                    
                    # filter dat only for this line
                    filt1 = data_file['LINE_NO'] == line
                    data_file_line = data_file[filt1]

                    # average the data
                    data_file_line_averaged = (data_file_line
                                .groupby(data_file_line.index // sounding_averaging)
                                .mean() 
                                .reset_index(drop=True) )

                    # append partial results
                    data_file_averaged_list.append(data_file_line_averaged)
                
                if data_file_averaged_list:
                    data_file_averaged = pd.concat(data_file_averaged_list, axis=0, ignore_index=True)
                else:
                    data_file_averaged = pd.DataFrame()
                return data_file_averaged

            data_file = average_data_file(data_file=data_file, sounding_averaging=self.sounding_averaging)



        # INDEX THE DATAFRAME

        # this includes the data and the uncertainty, 
        # this is split later 
        data_file_indexed = data_file.set_index(["LINE_NO", "TIME"])



        # SEPARATE DATA IN COLUMNS

        # WARNING: The following subset of columns can be filtered by 
        #          LM_X or LM_Z, 
        #          this needs to be checked.
        #          
        #          Also the DES file mentions this data has been processed and NORMALISED,
        #          and the data looks different from the condamine data.
        #          This need to be checked. 
        lm_cols = [x for x in list(data_file.columns) if 'LM_Z' in x.upper()]
        hm_cols = [x for x in list(data_file.columns) if "HM_Z" in x.upper()]

        # WARNING: Check if we need to return all the gates or only the valid ones.
        #          The filter gets applied here. 
        #          The flag to activate it or deactivate it can be changed to True/False
        filter_validgates_only = True
        if filter_validgates_only:
            lm_cols = [x for x in lm_cols if int(x.split('_')[-1]) in Survey.validgate_list_lm]
            hm_cols = [x for x in hm_cols if int(x.split("_")[-1]) in Survey.validgate_list_hm]

        lm_cols_dataonly = [x for x in lm_cols if 'RUNC' not in x.upper()]
        hm_cols_dataonly = [x for x in hm_cols if 'RUNC' not in x.upper()]

        lm_cols_unconly = [x for x in lm_cols if 'RUNC' in x.upper()]
        hm_cols_unconly = [x for x in hm_cols if 'RUNC' in x.upper()]
        


        # UNIT CONVERSION

        # convert pico volts to volts
        data_file_indexed[lm_cols_dataonly] = data_file_indexed[lm_cols_dataonly] * 1e-12
        data_file_indexed[hm_cols_dataonly] = data_file_indexed[hm_cols_dataonly] * 1e-12



        # STATION DATA

        station_data = data_file[["DATE",
                                  "TIME",
                                  "LINE_NO",
                                  "UTMX",
                                  "UTMY",
                                  "ELEVATION",
                                  "RX_ALTITUDE",
                                  "TX_ALTITUDE",
                                  "RX_ALTITUDE_STD",
                                  "TX_ALTITUDE_STD",  
                                  "AngleX",
                                  "AngleY"]]

        station_data = (station_data
                            .set_index(["LINE_NO", "TIME"])
                            .drop_duplicates() )



        # SAVE VARIABLES

        self.station_data = station_data

        self.lm_data = data_file_indexed[lm_cols_dataonly]
        self.hm_data = data_file_indexed[hm_cols_dataonly]

        self.lm_relunc = data_file_indexed[lm_cols_unconly]+self.runc_offset
        self.hm_relunc = data_file_indexed[hm_cols_unconly]+self.runc_offset

        self.lm_std = self.lm_relunc * self.lm_data.values
        self.hm_std = self.hm_relunc * self.hm_data.values

        self.lm_cols_dataonly = lm_cols_dataonly
        self.hm_cols_dataonly = hm_cols_dataonly

        return None

    def lm_data_(self):

        return self.station_data[self.lm_cols_dataonly]




class Survey:
    # default location for gex in r"data\condamine\20230616_10098_306HP_LM_MergeGates_HM_splinegates_final_Z_V3.gex"

    def __init__(self):
        self.sounding_averaging = None
        pass

    def proc_gex(self, gexloc):
        gex_file = parse_gex_file(gexloc)
        n_lm_gates = (
            gex_file["Channel1"]["NoGates"] - gex_file["Channel1"]["RemoveInitialGates"]
        )
        n_hm_gates = (
            gex_file["Channel2"]["NoGates"] - gex_file["Channel2"]["RemoveInitialGates"]
        )

        skipgates = gex_file["Channel1"]["RemoveInitialGates"]

        cols = list(gex_file["General"].keys())
        LMCols = list(filter(lambda k: "GateTimeLM" in k, cols))
        HMCols = list(filter(lambda k: "GateTimeHM" in k, cols))

        lm_gate_centers = [gex_file["General"][x][0] for x in LMCols]
        channel_info = gex_file["Channel1"]
        lm_shift = channel_info["GateTimeShift"]
        lm_delay = channel_info["MeaTimeDelay"]
        first_gate = channel_info["RemoveInitialGates"]
        last_gate = channel_info["NoGates"]
        lm_times = np.array(lm_gate_centers[first_gate:last_gate]) + lm_shift + lm_delay
        

        hm_gate_centers = [gex_file["General"][x][0] for x in HMCols]
        channel_info = gex_file["Channel2"]
        hm_shift = channel_info["GateTimeShift"]
        hm_delay = channel_info["MeaTimeDelay"]
        first_gate = channel_info["RemoveInitialGates"]
        last_gate = channel_info["NoGates"]
        hm_times = np.array(hm_gate_centers[first_gate:last_gate]) + hm_shift + hm_delay

        waves = gex_file["General"]["Waveforms"]
        lm_wave_time = waves["LM"]["time"]
        lm_wave_form = waves["LM"]["form"]
        hm_wave_time = waves["HM"]["time"]
        hm_wave_form = waves["HM"]["form"]

        tx_area = gex_file["General"]["TxLoopArea"]

        tx_shape = np.pad(gex_file["General"]["TxLoopPoints"], (0, 1))
        tx_shape[-1] = tx_shape[0]
        rx_offset = gex_file["General"]["RxCoilPosition1"]

        self.n_lm_gates = n_lm_gates
        self.n_hm_gates = n_hm_gates
        self.skipgates = skipgates
        self.lm_times = lm_times
        self.hm_times = hm_times
        self.tx_shape = tx_shape
        self.rx_offset = rx_offset
        self.lm_wave_time = lm_wave_time
        self.lm_wave_form = lm_wave_form
        self.hm_wave_time = hm_wave_time
        self.hm_wave_form = hm_wave_form
        self.tx_area = tx_area
        self.hm_shift = hm_shift
        self.lm_shift = lm_shift
        self.hm_delay = hm_delay
        self.lm_delay = lm_delay



    def proc_des(self, desloc):
        
        des_file = parse_des_file(desloc)

        df_validgates_lm = des_file['df_validgates_lm']
        df_validgates_hm = des_file['df_validgates_hm']

        validgate_list_lm = des_file['validgate_list_lm']
        validgate_list_hm = des_file["validgate_list_hm"]

        # WARNING: these two variables were present in the GEX file
        #          and read from the file, 
        #          but they aren't present in the DES file
        #          so we initialised them as zero
        shift_lm = 0
        delay_lm = 0

        df_validgates_lm['lm_times'] = df_validgates_lm['centre_us'] + shift_lm + delay_lm

        # us to s
        df_validgates_lm["lm_times"] = (df_validgates_lm["lm_times"] * 1e-6)



        # WARNING: these two variables were present in the GEX file
        #          and read from the file,
        #          but they aren't present in the DES file
        #          so we initialised them as zero
        shift_hm = 0
        delay_hm = 0

        df_validgates_hm["hm_times"] = df_validgates_hm["centre_us"] + shift_hm + delay_hm

        # us to s
        df_validgates_hm["hm_times"] = (df_validgates_hm["hm_times"] * 1e-6)
        


        tx_shape = des_file['df_loop_geom_tx']
        tx_shape['_'] = 0
        tx_shape = tx_shape.to_numpy()



        rx_offset = des_file["df_instrument_position"]
        rx_offset = rx_offset.loc['Z-coil', ['x_m', 'y_m', 'z_m']].astype(float).to_numpy()



        lm_wave_form = des_file['df_txwaveform_lm']['amplitude'].to_numpy()
        lm_wave_time = des_file['df_txwaveform_lm']['time_s'].to_numpy()


                    
        hm_wave_form = des_file['df_txwaveform_hm']['amplitude'].to_numpy()
        hm_wave_time = des_file['df_txwaveform_hm']['time_s'].to_numpy()



        def build_full_wave_form(in_times, in_wave_form, waveform_specs):
            
            df_wave_in = pd.DataFrame({'time':in_times,
                                       'amplitude':in_wave_form})



            on_time = waveform_specs.loc['tx on-time']

            on_time_items = on_time.split(' ')

            # ms to s
            if on_time_items[1] == 'ms':
                on_time = float(on_time_items[0]) / 1000
            else:
                on_time = float(on_time_items[0])
            


            off_time = waveform_specs.loc['tx off-time']

            off_time_items = off_time.split(' ')

            # ms to s
            if off_time_items[1] == 'ms':
                off_time = float(off_time_items[0]) / 1000
            else:
                off_time = float(off_time_items[0])


            # create new part of the wave
            filt1 = df_wave_in['time'] <= on_time + off_time
            df_wave_out = df_wave_in[filt1]

            # adjust time and amplitude of new part of the wave            
            df_wave_out['time'] = df_wave_out['time'] - on_time - off_time
            df_wave_out['amplitude'] = df_wave_out['amplitude'] * -1

            # add the new and existing parts of the wave
            df_wave_out = pd.concat([df_wave_out, df_wave_in], axis=0)

            # remove last point - to make it look like the example
            #df_wave_out = df_wave_out[:-1]

            # prep outputs
            out_times = df_wave_out['time'].to_numpy()
            out_wave_form = df_wave_out['amplitude'].to_numpy()

            return out_times, out_wave_form



        (lm_wave_time,
        lm_wave_form) = build_full_wave_form(in_times=lm_wave_time, 
                                            in_wave_form=lm_wave_form,
                                            waveform_specs=des_file['tx_waveform_specs']['lm'])
        
        (hm_wave_time,
        hm_wave_form) = build_full_wave_form(in_times=hm_wave_time, 
                                            in_wave_form=hm_wave_form,
                                            waveform_specs=des_file['tx_waveform_specs']['hm'])


        tx_area = des_file["transmitter_specs"]['Tx Loop Area']
        tx_area = tx_area.lower().strip('m2').strip()
        tx_area = float(tx_area)

        rx_area = des_file["receiver_specs"]['Rx coil effective area']
        rx_area = rx_area.split(' ')
        rx_area_dic = {rx_area[0].lower():float(rx_area[2]), rx_area[4].lower():float(rx_area[6])}




        self.n_lm_gates = des_file["n_lm_gates"]
        self.n_hm_gates = des_file["n_hm_gates"]
        # self.skipgates = skipgates      # not used when reading in this format
        self.col_name_list = des_file["col_name_list"]
        self.lm_times = df_validgates_lm["lm_times"].to_numpy()    
        self.hm_times = df_validgates_hm["hm_times"].to_numpy()
        self.tx_shape = tx_shape
        self.rx_offset = rx_offset
        self.lm_wave_form = lm_wave_form
        self.lm_wave_time = lm_wave_time
        self.hm_wave_time = hm_wave_time
        self.hm_wave_form = hm_wave_form
        self.tx_area = tx_area
        self.rx_area = rx_area_dic

        self.hm_shift = shift_hm
        self.lm_shift = shift_lm
        self.lm_delay = delay_lm
        self.hm_delay = delay_hm
        
        
        self.validgate_list_lm = validgate_list_lm
        self.validgate_list_hm = validgate_list_hm

        return None



    def add_data(self, DFILE):
        extension = DFILE.split(".")[-1].lower()

        if extension == "parquet":
            self.Data = Data()
            self.Data.proc_parquet(Survey=self, DFILE=DFILE)

        elif extension == "dat":
            self.Data = Data(sounding_averaging=self.sounding_averaging)
            self.Data.proc_dat(Survey=self, DFILE=DFILE)
            
        elif extension == "xyz":
            self.Data = Data(sounding_averaging=self.sounding_averaging)
            self.Data.proc_xyz(Survey=self, DFILE=DFILE)

    
    def line_list(self):
        '''Return the list of lines'''

        station_data = self.Data.station_data
        station_data_ = station_data.reset_index(drop=False)
        line_list = list(station_data_["LINE_NO"].unique())

        return line_list

    def time_list(self, line=None):
        """Return the list of lines"""

        station_data = self.Data.station_data
        station_data_ = station_data.reset_index(drop=False)

        if line:
            filt1 = station_data_['LINE_NO'] == line
            station_data_ = station_data_.loc[filt1]

        time_list = list(station_data_["TIME"].unique())

        return time_list



if __name__ == "__main__":

    testing = True

    if testing:
        
        import os

        fd_script = r"c:\Users\berrioj\OneDrive - ITP (Queensland Government)\Scripts\AEM_RML"

        os.chdir(fd_script)

        print(f"cwd: {os.getcwd()}")

        DFILE = r".\data\injune\AUS_10024_InJune_EM_MGA55.dat"  # data file (PARQUET, DAT, etc)
        IFILE = r".\data\injune\AUS_10024_InJune_EM_MGA55.des"  # system info file (GEX, DES, etc)
        
        survey = Survey()
        survey.proc_des(IFILE)

        survey.sounding_averaging = 3
        survey.add_data(DFILE)

        a1 = survey.Data.lm_data
        #a2 = survey.Data.lm_data_()

        print()
