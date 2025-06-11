import os
from energy_allocator import EnergyAllocator

# Path of PV production file
production_dir      = 'local_tests\production'
production_dir_path = os.path.relpath(production_dir)
production_file     = 'hirvikoirankatu_2_simulation_ver2.csv'
production_path     = os.path.join(production_dir_path, production_file)

# Path of housing company consumption file
company_dir         = 'local_tests\consumption_profiles'
company_dir_path    = os.path.relpath(company_dir)
company_file        = 'Taloyhtiö_kulutusprofiili_hirvikoirankatu_2.csv'
company_path        = os.path.join(company_dir_path, company_file)

# Path of apartments consumption files
apartment_dir       = company_dir
apartment_dir_path  = company_dir_path
apartment_1_file    = 'A1_kulutusprofiili_hirvikoirankatu_2.csv'
apartment_2_file    = 'B6_kulutusprofiili_hirvikoirankatu_2.csv'
apartment_3_file    = 'C10_kulutusprofiili_hirvikoirankatu_2.csv'
apartment_1_path    = os.path.join(apartment_dir_path, apartment_1_file)
apartment_2_path    = os.path.join(apartment_dir_path, apartment_2_file)
apartment_3_path    = os.path.join(apartment_dir_path, apartment_3_file)

# Test apartment consumption profiles and paths
test_apartment_dir      = 'local_tests/test_profiles'
test_apartment_dir_path = os.path.relpath(test_apartment_dir)
test_apartments_consumption_dict = {}
test_apartment_files    = os.scandir(test_apartment_dir)
for apartment in test_apartment_files:
    if apartment.name[2] == '_':
        test_apartments_consumption_dict[apartment.name[0:2]] = os.path.join(
            test_apartment_dir_path, apartment.name)
    else:
        test_apartments_consumption_dict[apartment.name[0:3]] = os.path.join(
            test_apartment_dir_path, apartment.name)


# apartment energy allocation dict
apartments_allocation_dict  = {
    'A1':0.06,
    'A2':0.06,
    'A3':0.06,
    'A4':0.06,
    'B5':0.09,
    'B6':0.09,
    'B7':0.09,
    'C8':0.12,
    'C9':0.12,
    'C10':0.12,
    'C11':0.12,
}
# appartment consumption dict
apartments_consumption_dict = {
    'A':apartment_1_path,
    'B':apartment_2_path,
    'C':apartment_3_path
}

allocator = EnergyAllocator(
    production_path, 
    company_path, 
    apartments_consumption_dict, 
    apartments_allocation_dict,
    )

enerloc_df = allocator.financial_value_sum()
print(enerloc_df)
with_energy_community       = enerloc_df.sum()
print("Total value with energy community [€]:", with_energy_community/100)

#enerloc_df.to_csv(os.path.join(os.path.relpath('local_tests/result_csv'), 'test.csv'))

