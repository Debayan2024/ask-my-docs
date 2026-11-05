print("Executing bootstrap.py: Attempting to patch sqlite3...")
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("Successfully patched sqlite3 with pysqlite3-binary via bootstrap.")
except ImportError:
    print("pysqlite3-binary not found during bootstrap. Using system sqlite3.")
    pass
print("Bootstrap.py execution finished.")