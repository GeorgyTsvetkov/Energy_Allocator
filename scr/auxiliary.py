date_time_formats = [
    "%m/%d/",               # mm/dd
    "%d/%m/",               # dd/mm
    "%m-%d",                # mm-dd
    "%f/%e/",               # m/d
    "%b %e, ",              # Mon d
    "%B %e, ",              # Month d
    "%m-%d %H:%M:%S",       # mm-dd hh:mm:ss
    "%m-%d %I:%M:%S %p",    # mm-dd hh:mm:ss AM/PM
    "%d %B ",               # dd Month
    "%b %e, ",              # Mon d,
    "%H:%M:%S",             # 24-hour time
    "%I:%M:%S %p",          # 12-hour time with AM/PM
    "%d.%m. %H:%M",         # PVSOL format    

    "%m/%d/%Y",             # mm/dd/yyyy
    "%d/%m/%Y",             # dd/mm/yyyy
    "%Y-%m-%d",             # yyyy-mm-dd
    "%f/%e/%Y",             # m/d/yyyy (platform-specific)
    "%b %e, %Y",            # Mon d, yyyy
    "%B %e, %Y",            # Month d, yyyy
    "%Y-%m-%d %H:%M:%S",    # yyyy-mm-dd hh:mm:ss
    "%Y-%m-%d %I:%M:%S %p", # yyyy-mm-dd hh:mm:ss AM/PM
    "%d %B %Y",             # dd Month yyyy
    "%b %e, %Y",            # Mon d, yyyy
    "%H:%M:%S",             # 24-hour time
    "%I:%M:%S %p",          # 12-hour time with AM/PM
    "%Y-%d.%m. %H:%M",      # PVSOL + year
    "%Y-%m-%d %H:%M:%S%z"   # Pandas DateTime ISO 8601 format with timezone
]