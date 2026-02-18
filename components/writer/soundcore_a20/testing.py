# Requires: pip install bleak asyncio

import asyncio
import logging
from bleak import BleakScanner, BleakClient
from bleak.exc import BleakError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Replace this name with the exact device name you see in your BLE scan.
TARGET_NAME = "soundcore Sleep A20"

async def scan_and_connect():
    try:
        logger.info("Starting BLE device scan...")
        devices = await BleakScanner.discover()
        
        logger.info(f"Found {len(devices)} BLE devices:")
        for d in devices:
            logger.info(f"Device: {d.name or 'Unknown'} - Address: {d.address}")
        
        target = None
        for d in devices:
            if d.name and TARGET_NAME in d.name:
                target = d
                logger.info(f"Found target device: {d.name} at {d.address}")
                break

        if not target:
            logger.error(f"No device named '{TARGET_NAME}' found.")
            return

        address = target.address
        logger.info(f"Attempting to connect to {TARGET_NAME} at {address}...")
        
        try:
            async with BleakClient(address, timeout=10.0) as client:
                if not client.is_connected:
                    logger.error("Failed to connect to device.")
                    return
                
                logger.info(f"Successfully connected to {TARGET_NAME} ({address})")
                
                # List all services and characteristics
                services = await client.get_services()
                logger.info(f"Found {len(services)} services:")
                
                for service in services:
                    logger.info(f"\n[Service] {service.uuid}")
                    logger.info(f"  Description: {service.description}")
                    
                    for char in service.characteristics:
                        props = ",".join(char.properties)
                        logger.info(f"  └─[Characteristic] {char.uuid}")
                        logger.info(f"     Properties: {props}")
                        
                        # Try to read the characteristic if it's readable
                        if "read" in char.properties:
                            try:
                                value = await client.read_gatt_char(char.uuid)
                                logger.info(f"     Value: {value.hex()}")
                            except BleakError as e:
                                logger.warning(f"     Could not read value: {str(e)}")
                
                logger.info("\nConnection test completed successfully!")
                
        except BleakError as e:
            logger.error(f"Connection error: {str(e)}")
            
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting BLE device testing...")
    asyncio.run(scan_and_connect())
    logger.info("Testing completed.")
