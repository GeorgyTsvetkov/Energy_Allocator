"""
TODO program get_spot_price(), align_prices_by_hour() and 
calculate_price_median() to use pandas.DataFrame.
"""

import numpy as np
import requests
from datetime import datetime
from collections import defaultdict
import pandas as pd
import traceback
from auxiliary import date_time_formats
import time

class EnergyAllocator:
    """
    Calculates amount and value of energy received by a housing company and
    apartments from an energy source (i.e. photovoltaic power plant),
    and the amount and value of energy leftover and sold to electricity grid
    operator.

    Attributes:
        production_path (str) : Path to CSV file with hourly energy production.
        company_path (str) :    Path to CSV file with hourly housing company 
                                consumption.
        app_consumption_dict (dict[str:str]) : Dictionary with house identifier or 
                                        appartment as key, and path to 
                                        corresponding CSV hourly consumption 
                                        profile file as value.
        app_allocation (dict[str:float]) : Dictionary with house identifier or
                                appartment as key, and path to corresponding 
                                CSV hourly consumption profile file as value.
        analysis_length (int) : Number of years of electricity spot price data
                                to analyze (default: 5).

    """
    # Class constants

    def __init__(
            self, 
            production_path         :str                = None,
            company_path            :str                = None,
            app_consumption_dict    :dict[str:str]      = None,
            app_allocation_dict     :dict[str:float]    = None,
            analysis_length         :int                = 5,
            vat_perc                :float              = 0.255,
            transfer_fee_perc       :float              = 0.111
    ):
        self.production_path        = production_path
        self.company_path           = company_path
        self.app_consumption_dict   = app_consumption_dict
        self.app_allocation_dict    = app_allocation_dict
        self.analysis_length        = analysis_length
        self.additional_fee         = (1 + vat_perc+transfer_fee_perc)


        self.spot_price_median              = 'Spot median [c/kWh]'
        self.spot_price_median_fees         = 'Spot median w/ fees [c/kWh]'
        self.production_column              = 'PV production [kWh]'
        self.company_column                 = 'Company consumption [kWh]'
        self.company_after_pv               = 'Company after PV [kWh]'
        self.value_after_subtraction        = 'Company value of coverage [c]'
        self.pv_after_company               = 'PV after company [kWh]'
        self.temp_apartment_consumption     = 'APP consumption [kWh]'
        self.temp_app_after_pv_column       = 'APP after PV [kWh]'
        self.temp_allocation_of_coverage    = 'APP value of coverage [c]'
        self.pv_over_production             = 'PV over production [kWh]'
        self.value_to_grid                  = 'Electricity value to grid [c]'

    
    def get_spot_price(self) -> list[dict]:
        """
        Fetches historical spot price data from the Finnish electricity 
        market API.

        Returns:
            List[dict]: List of hourly price records in the format:
                [{'date': str, 'value': float}], where 'value' is in snt/kWh.
        """

        start_year  = datetime.now().year-1
        start       = f"{start_year-(self.analysis_length-1)}-01-01T00:00:00.000Z"
        end         = f"{start_year}-12-31T23:00:00Z"
        url         = f"https://sahkotin.fi/prices?start={start}&end={end}"
        try:
            # Establish connection with API, raise for status if needed
            response        = requests.get(url)
            if response.raise_for_status() is not None:
                print(f"\nStatus: {response.raise_for_status()}\n")
            # Parse API response as JSON and extract values, price = snt/kWh
            spot            = response.json()
            date_price_list = [
                            {'date':entry['date'], 'value': entry['value'] * 0.1}
                            for entries in spot.values()
                            for entry in entries
            ]
            return date_price_list
        except Exception as e:
            print(f"[get_spot_prices] Error: {e}")
            traceback.print_exc()
            return None

    def align_prices_by_hour(self, data:list[dict]) -> list[list]:
        """
        Aligns hourly spot price data by calendar hour (month-day and hour) 
        across multiple years.

        Groups prices using the format '%m-%dT%H:%M:%S.000Z', allowing statistical
        comparison across years for each hour of the year.

        Args:
            data (List[dict]): Spot price data with datetime strings and values.

        Returns:
            List[List]: List of lists, each beginning with a time key, followed 
            by price values from different years.
        """

        # Dictionary to group prices by "MM-DDTHH:MM:SS.000Z"
        hour_to_prices  = defaultdict(list)
        try:
            for entry in data:
                # Parse datetime and create hour key
                dt  = datetime.strptime(entry["date"], "%Y-%m-%dT%H:%M:%S.000Z")
                hour_key   = dt.strftime("%m-%dT%H:%M:%S.000Z")
                # Append price for each hour
                hour_to_prices[hour_key].append(entry["value"])
            #Create final list with hour key + all prices
            result = [[hour] + prices for hour, prices in sorted(hour_to_prices.items())]
            return result
        except Exception as e:
            print(f"[align_prices_by_hour] Error: {e}")
            traceback.print_exc()
            return None

    def calculate_price_median(self, data:list[list]) -> list[list]:
        """
        Appends the median electricity price for each hour-of-year across all years.

        Modifies input list in-place, adding the median of the price values
        as the last element in each row.

        Args:
            data (List[List]): Hourly spot price data grouped by calendar hour.

        Returns:
            List[List]: Updated data with median price appended to each row.
        """
        try:
            for entry in data:
                prices  = entry[1:]
                entry.append(np.median(prices))
            return data
        except Exception as e:
            print(f"[calculate_price_median] Unable to calculate: {e}")
            return None
    
    def run_price_analysis(self) -> pd.DataFrame:
        """
        Performs full spot price analysis pipeline.

        Fetches and groups multi-year hourly spot prices, calculates the median price
        for each hour, and formats results into a datetime-indexed DataFrame for 2024.

        Returns:
            pd.DataFrame: DataFrame indexed by hourly timestamps (2024), 
            containing:
                - 'Spot median [c/kWh]'
                - 'Spot median w/ fees [c/kWh]'
        """
        try:
            raw_data        = self.get_spot_price()
            aligned_data    = self.align_prices_by_hour(raw_data)
            final_data      = self.calculate_price_median(aligned_data)
            dataframe       = pd.DataFrame(final_data)
            dataframe[0]    = pd.to_datetime('2024-' + dataframe[0], utc=True)
            dataframe.set_index(0, inplace=True)
            dataframe           = dataframe[[dataframe.columns[-1]]]
            dataframe.columns   = [*dataframe.columns[:-1], self.spot_price_median]
            dataframe[self.spot_price_median_fees] = (dataframe[self.spot_price_median] * self.additional_fee)
            return dataframe
        except Exception as e:
            print(f"[run_price_analysis] Error: {e}")
            traceback.print_exc()
            return None

    def remove_feb_29_if_mismatch(self, df1: pd.DataFrame, df2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        """
        Removes February 29th from one DataFrame if the other contains only 
        8760 hourly rows (non-leap year), to align them for time series operations.

        Args:
            df1 (pd.DataFrame): First time-indexed DataFrame.
            df2 (pd.DataFrame): Second time-indexed DataFrame.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame] | None: Aligned DataFrames with leap day 
            removed from one if needed, or None on error.
        """
        try:
            len1, len2 = len(df1), len(df2)

            # Define helper to remove Feb 29
            def remove_feb_29(df: pd.DataFrame) -> pd.DataFrame:
                return df[~((df.index.month == 2) & (df.index.day == 29))]

            # Only remove Feb 29 from one of the DataFrames if lengths mismatch
            if len1 == 8784 and len2 == 8760:
                df1 = remove_feb_29(df1)
            elif len2 == 8784 and len1 == 8760:
                df2 = remove_feb_29(df2)
            elif len1 != len2:
                print(f"[Warning] Length mismatch but unexpected row counts: df1={len1}, df2={len2}")

            return df1, df2

        except Exception as e:
            print(f"[remove_feb_29_if_mismatch] Error: {e}")
            traceback.print_exc()
            return None
    
    def to_dataframe_converter(self, path:str) -> pd.DataFrame:
        """
        Converts a CSV file to dataframe. Checks if format of first column
        corresponds to a known date-time format. Converts to datetime index
        according to detected format.

        Args:
            path (str): Path to CSV file to convert dataframe

        Returns:
            pd.DataFrame: DataFrame with a DateTime index column and value 
            column

        """
        try:
            dataframe   = pd.read_csv(path, header=None)
        except FileNotFoundError:
            print(f"[to_dataframe_converter] File not found: {path}")
            return None
        matched_format = None
        try:
            for sample in dataframe[0].dropna().astype(str).head(20):
                for dt_format in date_time_formats:
                    try:
                       time.strptime(sample.strip(), dt_format)
                       matched_format   = dt_format
                       break
                    except ValueError:
                        continue
                if matched_format:
                    break
            if matched_format:
                if "Y" in  matched_format or "y" in matched_format:
                    dataframe[0]       = pd.to_datetime(
                        dataframe[0], 
                        format=matched_format, 
                        utc=True)
                else:
                    dataframe[0]    = pd.to_datetime(
                                '2024-' + dataframe[0],
                                format=f'%Y-{matched_format}', 
                                utc=True
                                )
                dataframe.set_index(0, inplace=True)
            return dataframe
        except Exception as e:
            print(f"[to_dataframe_converter] Error : {e}")
            traceback.print_exc()
            return None
        
    def add_production(self) -> pd.DataFrame:
        """
        Loads PV production data from CSV, aligns it with hourly spot price index,
        and appends it to the spot price DataFrame.

        Returns:
            pd.DataFrame: DataFrame with 'PV production [kWh]' added, indexed by datetime.
        """

        try:
            if self.production_path is not None:
                self.production_df  = self.to_dataframe_converter(self.production_path)
                self.production_df.columns = [self.production_column]
                dataframe = self.run_price_analysis()
                # Ensure datetime index for both dataframes
                if not isinstance(self.production_df.index, pd.DatetimeIndex):
                    self.production_df.index   = pd.to_datetime(
                        self.production_df.index, utc=True)
                # Drop Feb 29 if one dataframe has 8760 hourse (leap year mismatch)
                self.production_df, dataframe = self.remove_feb_29_if_mismatch(self.production_df, dataframe)
                # Check for specified column in dataframe
                if self.production_column not in self.production_df.columns:
                    raise KeyError(f"Column '{self.production_column}' not found in production data.")
                # Align PV production index to match dataframe
                self.production_df = self.production_df.reindex(dataframe.index)
                # Check for NaN values in value or index columns respectively
                if self.production_df[self.production_column].isna().any():
                    print("[add_production] Warning: NaN values after reindexing production data.")
                if not self.production_df.index.difference(dataframe.index).empty:
                    print("[add_production] Warning: Production index partially mismatched with price index.")
                # Assign aligned PV production to new column
                dataframe[self.production_column] = self.production_df[self.production_column]
                return dataframe
            else:
                return None
        except Exception as e:
            print(f"[add_production] Error : {e}")
            traceback.print_exc()
            return None

    def add_company_consumption(self) -> pd.DataFrame:
        """
        Loads housing company consumption data, aligns it to the hourly index
        of the spot price (and PV data, if present), and appends it to the DataFrame.

        Returns:
            pd.DataFrame: DataFrame with 'Company consumption [kWh]' added.
        """

        try:
            company_df  = self.to_dataframe_converter(self.company_path)
            company_df.columns = [self.company_column]
            # Determine function to use as basis for dataframe
            dataframe = self.add_production() if self.production_path is not None else self.run_price_analysis()
            # Ensure datetime index for both dataframes
            if not isinstance(company_df.index, pd.DatetimeIndex):
                company_df.index = pd.to_datetime(company_df.index, utc=True)
            # Drop Feb 29 if one dataframe has 8760 hourse (leap year mismatch)
            company_df, dataframe = self.remove_feb_29_if_mismatch(company_df, dataframe)
            # Check for specified column in dataframe
            if self.company_column not in company_df.columns:
                raise KeyError(f"Column '{self.company_column}' not found in production data.")
            # Align company_df consumption index to match dataframe
            company_df = company_df.reindex(dataframe.index)
            # Check for NaN values in value or index columns respectively
            if company_df[self.company_column].isna().any():
                print("[add_company_consumption] Warning: NaN values after reindexing company_df consumption data.")
            if company_df.index.difference(dataframe.index).empty:
                print("[add_production] Consumption index partially mismatched with price index.")
            # Assign aligned company_df consumption to new column
            dataframe[self.company_column] = company_df[self.company_column]
            return dataframe
        except Exception as e:
            print(f"[add_company_df_consumption] Error: {e}")
            traceback.print_exc()
            return None
        
    def add_apartment_consumption(self) -> pd.DataFrame | None:
        """
        Constructs a DataFrame of hourly apartment consumption values
        based on assigned profiles and allocation dictionary.

        Returns:
            pd.DataFrame: DataFrame with apartment names as columns and consumption
            values [kWh] as values, indexed by datetime.
        """

        try:
            # Create a dictionary with consumption profiles matching house index
            profile_data    = {}
            for key, path in self.app_consumption_dict.items():
                df  = self.to_dataframe_converter(path)
                profile_data[key]   = df[1]
            # Create a dataframe matching consumption profile to house
            self.apartments_df  = pd.DataFrame()
            for apartment, allocation in self.app_allocation_dict.items():
                self.apartment_consumption  = self.temp_apartment_consumption.replace('APP', apartment)
                if len(self.app_consumption_dict) != len(self.app_allocation_dict):    
                    profile_key = apartment[0]
                elif len(self.app_consumption_dict) == len(self.app_allocation_dict):
                    profile_key = apartment
                self.apartments_df[self.apartment_consumption] = profile_data[profile_key]
            return self.apartments_df
        except Exception as e:
            print(f"[create_apartment_dataframe] Error in process: {e}")
            return None

    def calculate_company(self) -> pd.DataFrame:
        """
        Calculates hourly energy and value metrics for the housing company:
            - Unmet consumption after PV coverage.
            - Value of covered consumption (saved electricity cost).
            - PV energy remaining after company needs.

        Returns:
            pd.DataFrame: DataFrame with the following columns added:
                - 'Company after PV [kWh]'
                - 'Company coverage value [c]'
                - 'PV after company [kWh]'
        """

        try:
            dataframe               = self.add_company_consumption()
            # Safety check
            required_columns = [
                self.company_column, self.production_column, self.spot_price_median
            ]
            for col in required_columns:
                if col not in dataframe.columns:
                    raise KeyError(f"Column '{col}' missing from dataframe.")
            # Energy coverage calculation
            unmet_consumption = dataframe[self.company_column] - dataframe[self.production_column]
            dataframe[self.company_after_pv] = unmet_consumption.clip(lower=0)
            # Financial value calculation
            covered_consumption = dataframe[self.company_column] - dataframe[self.company_after_pv]
            value = (dataframe[self.spot_price_median_fees]) * covered_consumption
            dataframe[self.value_after_subtraction]  = value
            # Production leftover calculation
            excess_pv = dataframe[self.production_column] - dataframe[self.company_column]
            dataframe[self.pv_after_company] = excess_pv.clip(lower=0)
            return dataframe
        except Exception as e:
            print(f"[calculate_company] Error: {e}")
            traceback.print_exc()
            return None
        
    def concatenate_dataframes(self) -> pd.DataFrame:
        """
        Appends apartment-level hourly consumption profiles to the main DataFrame
        containing company and PV production data.

        Returns:
            pd.DataFrame: Combined DataFrame with apartment consumption columns added.
        """

        try:
            dataframe       = self.calculate_company()
            apartment_df    = self.add_apartment_consumption()
            # Ensure datetime index for both dataframes
            if not isinstance(apartment_df.index, pd.DatetimeIndex):
                apartment_df.index = pd.to_datetime(apartment_df.index, utc=True)
            # Drop Feb 29 if one dataframe has 8760 hourse (leap year mismatch)
            apartment_df, dataframe = self.remove_feb_29_if_mismatch(apartment_df, dataframe)
            # Align apartment_df consumption index to match dataframe
            apartment_df = apartment_df.reindex(dataframe.index)
            # Check for NaN values in value or index columns respectively
            if apartment_df.isna().any().any():
                print("[add_apartment_df_consumption] Warning: NaN values after reindexing apartment_df consumption data.")
            if apartment_df.index.difference(dataframe.index).empty:
                print("[add_production] Consumption index partially mismatched with price index.")
            for column in apartment_df.columns:
                dataframe[column] = apartment_df[column]
            return dataframe
        except Exception as e:
            print(f"[concatenate_dataframes] Error: {e}")
            traceback.print_exc()
            return None
        
    def calculate_apartment(self) -> pd.DataFrame:
        """
        Calculates hourly PV coverage and financial value for each apartment,
        based on the share of PV production remaining after the company's usage.

        Returns:
            pd.DataFrame: DataFrame with apartment PV coverage and value columns added.
        """

        try:
            dataframe   = self.concatenate_dataframes()
            for apartment, allocation in self.app_allocation_dict.items():
                self.app_after_pv_column    = self.temp_app_after_pv_column.replace('APP', apartment)
                self.allocation_of_coverage = self.temp_allocation_of_coverage.replace('APP', apartment)

                app_after_pv:pd.DataFrame = dataframe[self.apartment_consumption] - (dataframe[self.pv_after_company]*allocation)
                dataframe.insert((dataframe.columns.get_loc(self.apartment_consumption)-1), self.app_after_pv_column, app_after_pv.clip(lower=0))
                covered_consumption = dataframe[self.apartment_consumption] - dataframe[self.app_after_pv_column]
                allocation_value = (dataframe['Spot median w/ fees [c/kWh]']) * covered_consumption
                dataframe.insert((dataframe.columns.get_loc(self.app_after_pv_column)+1), self.allocation_of_coverage, allocation_value)

            return dataframe
        except Exception as e:
            print(f"[calculate_apartment] Error: {e}")
            traceback.print_exc()
            return None

    def calculate_pv_over_production(self) -> pd.DataFrame:
        """
        Computes PV over-production after both company and apartment consumption,
        and its corresponding financial value (based on spot prices).

        Returns:
            pd.DataFrame: DataFrame with:
                - 'PV over production [kWh]'
                - 'Electricity value to grid [c]'
        """

        try:
            dataframe   = self.calculate_apartment()
            calculation = dataframe[self.production_column] - (dataframe[self.company_column] + sum([dataframe[column] for column in self.apartments_df]))
            dataframe[self.pv_over_production] = calculation.clip(lower=0)
            value   = dataframe[self.pv_over_production] * dataframe[self.spot_price_median]
            dataframe[self.value_to_grid]   = value
            return dataframe.round(3)
        except Exception as e:
            print(f"[calculate_pv_over_production] Error: {e}")
            traceback.print_exc()
            return None
    
    def financial_value_sum(self) -> pd.Series:  
        """
        Sums the total financial value (in cents) of:
            - Housing company PV coverage.
            - Apartment PV coverage.
            - PV over-production sold to the grid.

        Returns:
            pd.Series: Total values for all energy cost savings and grid sales [cents].
        """

        try:
            dataframe = self.calculate_pv_over_production()
            matching_columns = [col for col in dataframe.columns if "value" in col.lower()]
            return dataframe[matching_columns].sum().round(0)
        except Exception as e:
            print(f"[display_value_sum] Error: {e}")
            traceback.print_exc()
            return None
        
    def energy_value_sum(self) -> pd.Series:  
        """
        Sums the total energy value (in kWh) of:
            - Housing company PV coverage.
            - Apartment PV coverage.
            - PV over-production sold to the grid.

        Returns:
            pd.Series: Total values for all energy cost savings and grid sales [cents].
        """

        try:
            dataframe = self.calculate_pv_over_production()
            matching_after_columns = [col for col in dataframe.columns if "after PV" in col.lower()]
            return dataframe[matching_after_columns].sum().round(0)
        except Exception as e:
            print(f"[display_value_sum] Error: {e}")
            traceback.print_exc()
            return None