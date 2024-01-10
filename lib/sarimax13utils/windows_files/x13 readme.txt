1. Copy x13as.exe file and tools folder to c:\windows\system32
2. If you get an error on can't execute the program you should increase the paging size variable.
	It's a windows error. 
	Go Control Panel->System->Advanced setting->Performance->Settings->Set paging size at appropriate level >1Gb

copied from "C:\Users\EPostolit\AppData\Roaming\Mathematica\Applications\Economica\timeseries"

if the code breaks in windows with error


inside x13.py arima code (EPostolit/AppData/Local/Programs/Python/Python319/Lib/site-packages/statsmodels/tsa/x13.py)
replace 
"ISO-8859-1"



def _open_and_read(fname):
    # opens a file, reads it, and make sure it's closed
    with open(fname, 'r', encoding="utf-8") as fin:
        fout = fin.read()
    return fout

with 

def _open_and_read(fname):
    # opens a file, reads it, and make sure it's closed
    try:
        with open(fname, 'r', encoding="utf-8") as fin:
            fout = fin.read()
    except UnicodeDecodeError:
        with open(fname, 'r', encoding="ISO-8859-1") as fin:
            fout = fin.read()
    return fout

// change encoding in NamedTemporaryFile failed