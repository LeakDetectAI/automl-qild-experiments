import inspect
import logging
import os
from experiments.utils import setup_logging
from autoqild import *
import subprocess

DIR_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

if __name__ == "__main__":
    log_path = os.path.join(DIR_PATH, EXPERIMENTS, 'list_files.log')
    setup_logging(log_path=log_path)
    logger = logging.getLogger('Experiment')
    # Define the command to run
    command = "du -shc /scratch/hpc-prf-aiafs/prithag/deep-learning-sca/*"

    # Execute the command and capture its output
    try:
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                check=True)

        # Print the standard output (result.stdout)
        logger.info("Command Output:")
        logger.info(result.stdout)

        # Print the standard error (if any)
        if result.stderr:
            logger.info("\nError Output:")
            logger.info(result.stderr)
    except subprocess.CalledProcessError as e:
        logger.info("Error executing the command:")
        logger.info(e.stderr)

    # Define the command to run
    command = "du -shc /scratch/hpc-prf-aiafs/prithag/automl_qild_experiments/*"

    # Execute the command and capture its output
    try:
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                check=True)

        # Print the standard output (result.stdout)
        logger.info("Command Output:")
        logger.info(result.stdout)

        # Print the standard error (if any)
        if result.stderr:
            logger.info("\nError Output:")
            logger.info(result.stderr)
    except subprocess.CalledProcessError as e:
        logger.info("Error executing the command:")
        logger.info(e.stderr)
