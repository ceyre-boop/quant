"""
Warp Drive Benchmark — Massive Sample Test (10 Years)
Measures the velocity of the V5.2 Intelligence Substrate.
"""
import time
import logging
import pandas as pd
from training.feature_generator import FeatureGenerator

# Setup logging to capture timing only
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("WarpDrive")

def run_benchmark():
    gen = FeatureGenerator()
    
    logger.info("="*50)
    logger.info("STARSHIP SOVEREIGN: WARP DRIVE BENCHMARK")
    logger.info("Target: 50 Assets | 2,520 Bars (10 Years) | JIT Hurst")
    logger.info("="*50)

    # 1. MEASURE HARVEST (IO)
    start_io = time.time()
    all_data = gen.client.get_all_assets(timeframe='1Day', days=2520)
    io_time = time.time() - start_io
    total_bars = sum(len(df) for df in all_data.values())
    
    logger.info(f"IO HARVEST: {io_time:.2f}s | {total_bars:,} bars | {total_bars/io_time:.0f} bars/sec")

    # 2. MEASURE SYNTHESIS (CPU)
    start_cpu = time.time()
    # We simulate the processing of all assets
    from joblib import Parallel, delayed
    results = Parallel(n_jobs=gen.n_jobs)(
        delayed(gen._process_symbol)(symbol, df, all_data) 
        for symbol, df in all_data.items() if not df.empty and len(df) >= 50
    )
    combined = pd.concat(results, ignore_index=True)
    cpu_time = time.time() - start_cpu
    
    logger.info(f"CPU SYNTHESIS: {cpu_time:.2f}s | {len(combined):,} features | {len(combined)/cpu_time:.0f} features/sec")

    # 3. MEASURE PERSIST (DISK)
    start_disk = time.time()
    gen.save_dataset(combined, 'benchmark_10yr.parquet')
    disk_time = time.time() - start_disk
    
    logger.info(f"DISK PERSIST: {disk_time:.2f}s | {os.path.getsize('data/processed/benchmark_10yr.parquet')/1024/1024:.2f} MB")

    logger.info("="*50)
    logger.info(f"TOTAL RUNTIME: {io_time + cpu_time + disk_time:.2f}s")
    logger.info("="*50)

if __name__ == "__main__":
    import os
    run_benchmark()
