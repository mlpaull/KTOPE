# -*- coding: utf-8 -*-
"""
IMUNE processing for use with KTOPE

MIT License

Copyright (c) 2019 MICHAEL LOUIS PAULL

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Coded for Windows, using Python 3.6 64-bit
"""

import os
import time
import subprocess

class Sample:
    """Container for various characteristics of a sample
    
       Useful for referencing data for a specific sample.
       Is printed in the order project -> disease -> name
       
       Attributes:
           name (str): name given to sample
           classification (str): classification which sample belongs to
               e.g. control
           project (str): name of overall sample group e.g. name of a disease
    """
    
    def __init__(self,name,classification,project):
        self.name = name
        self.classification = classification
        self.project = project
        
    def __repr__(self):
        """Prints out basic details of Sample """
        s = self.project + ' ' + self.classification + ' '
        s += self.name
        return s
        
    def __eq__(self,other):
        """Tests whether two objects have the same attributes """
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __ne__(self,other):
        """!= is just the opposite of == """
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented
        
    def __hash__(self):
        """Object needs to have hash method to be used by set operator"""
        return hash(tuple(sorted(self.__dict__.items())))       
    
    @classmethod    
    def parse_sampleline(cls,sampleline):
        """Takes a line of text and creates Sample from it
        
           Samples can be written in multiple forms.
           (1)Project Classification Name
           (2)Sample: Project Classification Name
           (3)Project Classification Name FASTQ
           This function parses any of them into the correct format
           
           Args:
               sampleline (str): contains information to create sample
           Returns:
               sample (Sample): contains information from sampleline
        """
        attributes = sampleline.split()
        #If it starts with Sample:, remove this first part of the information
        if attributes[0] == 'Sample:':
            attributes = attributes[1:]
        #If there are more than 3 attributes, 4th is FASTQ file, so remove it
        if len(attributes) > 3:
            attributes = attributes[:-1]
        project = attributes[0]
        classification = attributes[1]
        name = attributes[2]
        sample = cls(name,classification,project)
        return sample


class IMUNE_processor:
    """Processes raw DNA or peptide data to determine kmers
    
       Has a method to run IMUNE_processor to take NGS data or peptide data
       and format into subsequences. Then it uses calculate-patterns.exe which 
       identifies kmers that meet criteria put in the created command file. 
        
       Attributes:
           rootdirectory (str): directory in which executable, Samples, and 
                Results folder are
           timings ([float]): contains times for runs
           errors ([str]): contains errors for runs
           stdout ([str]): contains output for runs
           processcmdfile (str): full filename of cmd file for processing
           kmercmdfile (str): full filename of cmd file for kmer calc
           samplelist [[str,str,str]]: list of lists with sample 
               information. Each element in list is 
               [Project,Classification,Name] for each sample in the order
               based on filelist
           cmddir (str): name of Commands directory
    """
    
    def __init__(self,rootdirectory):
        self.rootdirectory = rootdirectory
        self.timings = []
        self.errors = []
        self.stdout = []
        self.processcmdfile = ''
        self.kmercmdfile = ''
        self.samplelist = ''
        self.cmddir = os.path.join(self.rootdirectory,'Commands')
        
    def create_folders(self):
        """Creates folder structure used for kmer tiling data"""
        foldernames = ['Commands','logs','Samples','Results','Proteins',
                       'Data']
        for folder in foldernames:
            newfolder = os.path.join(self.rootdirectory,folder)
            os.mkdir(newfolder)  
            
    def initialize_samplelist(self,samplefile):
        """Initializes samplelist attribute using a .txt file
        
           Usually samplelist is initialized using the cmd_gen functions, but
           one may want to run the calculate_kmers on data that has already
           been processed into subsequences. The samplefile is a .txt file 
           where each line is Project Classification Name.
           
           Args:
               samplefile (str): filename of .txt file with sample info on 
                   on each line
           Returns:
               N/A
           
        """
        samplelist = []
        with open(samplefile,'r') as f:
            for line in f:
                sample = Sample.parse_sampleline(line)
                samplelist.append(sample)
        self.samplelist = samplelist
        
    def processor_NGS_cmd_gen(self,commandname,annealdict,datalabel,minseqlen,
                              maxseqlen):
        """Generates a command file for use with IMUNEprocessor program
        
           IMUNEprocessor uses a command file input to specify parameters. This
           function simplifies writing the command file by only taking in a few 
           parameters. datalabel is a .txt file which has "Project
           Classification Name Filename" on each line. 
           Outputs the filename of the command file.
           Intended for data in which sequences contain DNA coding for a 
           random peptide that is flanked by known annealing sequences.
           The data must have already been de-multiplexed. 
           
           Args:
               commandname (str): name of command file to be writen
               annealdict ({str}=str): contains annealing regions to look for 
                  in NGS data. 'before' key comes before sequence and 'after' 
                  sequence comes after sequence.
                  e.g. {'before':'GGCCAGTCTGGCCAGGGTGGA',
                        'after':'GGAGGGCAGTCTGGGCAG'}
               datalabel (str): filename of tab-delimited txt file with sample 
                   information. One line for each FASTQ file with the format:
                   "Project Classification Name Filename". One sample can come 
                   from multiple FASTQ files.
               minseqlen (int): Minimum length of peptide sequences in data
               maxseqlen (int): Maximum length of peptide sequences in data
           Returns:
               N/A
        """
        sampleinfo = []
        samplelist = []
        with open(datalabel,'r') as f:
            for line in f:
                #skip lines that are only newlines
                if line != '\n':
                    items = line.split()
                    if len(items) != 4:
                        raise ValueError('datalabel file is not formatted '
                                          'properly')
                    newsample = Sample.parse_sampleline(line)
                    #Get rid of newlines that may appear at end
                    filename = line.split()[3].strip()
                    if newsample not in samplelist:
                        samplelist.append(newsample)
                    #Append a tuple of sample and filename
                    sampleinfo.append((newsample,filename))
        processcmdfile = os.path.join(self.cmddir,commandname + '_cmd.txt')
        cmdtext = 'Processing Commands\n'
        cmdtext += 'Name: ' + commandname + '\n'
        cmdtext += 'Combine sequences with no more than 3 mutations\n\n'
        #print sample declarations
        for i,sample in enumerate(samplelist):
            cmdtext += 'Sample Declaration\n'
            cmdtext += 'Name: ' + sample.name + '\n'
            cmdtext += 'Project: ' + sample.project + '\n'
            cmdtext += 'Classification: ' + sample.classification + '\n'
            if i == 0:
                cmdtext += 'Sequence Type: Amino Acid\n'
                cmdtext += 'Minimum Sequence Length: ' + str(minseqlen) + '\n'
                cmdtext += 'Maximum Sequence Length: ' + str(maxseqlen) + '\n'
            cmdtext += '\n'
        #print fastq file locations
        for sampletuple in sampleinfo:
            sample,filename = sampletuple
            location,dataname = os.path.split(filename)
            cmdtext += 'File Declaration\n'
            cmdtext += 'Name: ' + dataname + '\n'
            cmdtext += 'Location: ' + location + '\n'   
            cmdtext += 'Sample ' + sample.project + ' '
            cmdtext += sample.classification + ' ' + sample.name            
            cmdtext += '\n'
            cmdtext += 'use the identified sequences ' + '\n'
            cmdtext += 'Construct Components: annealing sequence annealing'
            cmdtext += '\n'
            cmdtext += 'Component Sequences: ' + annealdict['before'] 
            cmdtext += ' sequence ' + annealdict['after'] + '\n\n'
        #Overwrite previous command file with same name
        if os.path.isfile(processcmdfile):
            os.remove(processcmdfile)       
        with open(processcmdfile,'w') as w:
            w.write(cmdtext)
        self.processcmdfile = processcmdfile
        self.samplelist = samplelist

    def processor_peptide_file_combine(self,filelist,combinedfile):
        """Combines multiple peptide lists for different samples into 1 file
        
           IMUNE_processor can process peptide lists if multiple samples are in
           1 file in the specified format. Each peptide list file should have a 
           first line with Project Classification Name. Each field should have
           no spaces. Every other line is a peptide found for that sample, 
           with each peptide only appearing once. Be careful with monitoring 
           RAM, if too many peptides from too many samples are combined, 
           RAM usage could become large.
           
           Args:
               filelist ([str]): list of file names of each sample's peptide 
                    lists
               combinedfile (str): filename of combined file created by
                    function
           Returns:
               N/A
        """
        peptidedict = {}
        samplelist = []
        for samplenum,filename in enumerate(filelist):
            with open(filename,'r') as f:
                firstline = next(f)
                sample = Sample.parse_sampleline(firstline)
                samplelist.append(sample)
                for line in f:
                    peptide = line.strip()
                    if not peptidedict.get(peptide,False):
                        peptidedict[peptide] = []
                    peptidedict[peptide].append(samplenum)
        with open(combinedfile,'w') as w:
            sampleline = 'Sequences'
            for sample in samplelist:
                sampleline += ' ' + sample.name
            w.write(sampleline + '\n')
            for peptide in peptidedict.keys():
                presence = [0]*len(samplelist)
                for samplenum in peptidedict[peptide]:
                    presence[samplenum] = 1
                presence =  ' '.join([str(i) for i in presence])
                w.write(peptide + ' ' + presence + '\n')
        self.samplelist = samplelist
             
    def processor_peptide_cmd_gen(self,combinedfile,commandname,minseqlen,
                                  maxseqlen):
        """Makes a command file for a list for a combined list of peptides
        
           Intended to run after processor_peptide_file_combine is called. 
           Takes information from combinedfile and arguments to make a command 
           file which can then be directly input into imune_processor or used 
           with cmdline_run.
           Note: samples in samplelist must be in the same order as they
           are in combinedfile
           
           Args:
               combinedfile (str): filename of combined file of peptides
               commandname (str): name of command file to be writen  
               minseqlen (int): Minimum length of peptide sequences in data
               maxseqlen (int): Maximum length of peptide sequences in data
           Returns:
               N/A
        """
        processcmdfile = os.path.join(self.cmddir,commandname + '_cmd.txt')
        cmdtext = 'Processing Commands\n'
        cmdtext += 'Name: ' + commandname + '\n'
        cmdtext += 'Combine sequences with no more than 0 mutations\n\n'
        #print sample declarations
        for i,sample in enumerate(self.samplelist):
            cmdtext += 'Sample Declaration\n'
            cmdtext += 'Name: ' + sample.name + '\n'
            cmdtext += 'Project: ' + sample.project + '\n'
            cmdtext += 'Classification: ' + sample.classification + '\n'
            if i == 0:
                cmdtext += 'Sequence Type: Amino Acid\n'
                cmdtext += 'Minimum Sequence Length: ' + str(minseqlen) + '\n'
                cmdtext += 'Maximum Sequence Length: ' + str(maxseqlen) + '\n'
            cmdtext += '\n'
        cmdtext += 'Reprocessing Declaration\n'
        location,dataname = os.path.split(combinedfile)
        cmdtext += 'Name: ' + dataname + '\n'
        cmdtext += 'Location: ' + location + '\n'
        for sample in self.samplelist:
            cmdtext += 'Sample ' + str(sample) + '\n'
        #Overwrite previous command file with same name
        if os.path.isfile(processcmdfile):
            os.remove(processcmdfile)       
        with open(processcmdfile,'w') as w:
            w.write(cmdtext)
        self.processcmdfile = processcmdfile
    
    @classmethod    
    def load_command(cls,rootdirectory,cmdfile,process=True):
        """Creates and returns imuneprocessor using already created cmdfile""" 
        imuneprocessor = cls(rootdirectory)
        if process:
            imuneprocessor.processcmdfile = cmdfile
        else:
            imuneprocessor.kmercmdfile = cmdfile
        return imuneprocessor
    
    def processing_cmdline_run(self,memory,maxthreads=None):
        """Runs IMUNEprocessor program using a command file
        
           Uses previously generated command file and subprocess module to 
           run command line. Returns information that would have been
           returned if command was run directly on command line. May be 
           preferable to run the command directly on command line to avoid 
           limitations of python. Progress can be viewed in the 
           'Calculations Summary' file in the Results\Processing\processcmdfile 
           folder or in the imuneprocessor.log file in logs.
           
           WARNING: ONCE THE PROCESS IS STARTED, IT CANNOT BE TERMINATED 
           THROUGH PYTHON. THE PROCESS HAS TO BE TERMINATED USING TASK MANAGER 
           AND THE RESULTS FOLDER CREATED IN SAMPLES SHOULD BE DELETED. 
           IT MAY BE NECESSARY TO QUIT THE PYTHON KERNEL. MAKE SURE YOU HAVE
           A 64-BIT VERSION OF JAVA.
           
           Args:
               memory (int): number of GB RAM to use
               maxthreads (int): maximum number of threads to use for
                   default is number of logical cores / 2 if not specificed
           Returns:
               N/A
        """
        os.chdir(self.rootdirectory)
        command = 'java -Xmx' + str(memory) + 'g'
        command += ' -jar imune-processor.jar ' + self.processcmdfile   
        if maxthreads:
            command += ' maxthreads ' + str(maxthreads)
        t1 = time.time()
        #Use subprocess.run to use command line arguments
        p = subprocess.run(command,stderr=subprocess.PIPE,
                          stdout=subprocess.PIPE,universal_newlines=True,
                          shell=True)
        timing = time.time()-t1
        errors = p.stderr
        stdout = p.stdout
        self.timings.append(timing)
        self.errors.append(errors)
        self.stdout.append(stdout)    
        if errors:
            for line in errors.split('\n'):
                print(line)
            raise RuntimeError('There were errors in kmer calculation')        
     
    def calculate_kmers_cmd_gen(self,samplefolder,outputfolder,mindef,
                                   maxdef,minlen,maxlen,enrichmin):
       """Make command file for kmer calculation based on parameters
       
          Writes a command file with all of the arguments of the function.
          Overwrites previous command file. If command is run for which results
          folder already exists, there will be an error.
          
          Args:
              outputfolder (str): name of folder to put run in. Should be
                  Projectname/Runname e.g. 'Disease\Run1'
              mindef (str): minimum defined characters for kmers
              maxdef (str): maximum defined characters for kmers
              minlen (str): minimum total characters for kmers
              maxlen (str): maximum total characters for kmers
              enrichmin (float): minimum enrichment threshold
          Returns:
              N/A
       """
       os.chdir(self.rootdirectory)
       text = 'Samples Folder: ' + samplefolder + '\n\n'
       text += 'Output Folder: Results' + os.sep + outputfolder + '\n\n'
       text += 'Minimum Defined Pattern Characters: ' + str(mindef) + '\n'
       text += 'Maximum Defined Pattern Characters: ' + str(maxdef) + '\n'
       text += 'Minimum Pattern Length: ' + str(minlen) + '\n'
       text += 'Maximum Pattern Length: ' + str(maxlen) + '\n\n'
       text += 'Filter: Enrichment' + '\n'
       text += 'Enrichment Min: ' + str(enrichmin) + '\n'
       text += 'Calculate Poisson Probability: False' + '\n'
       text += '-log(Poisson Probability) Min: 4.0' + '\n'
       text += 'Prevalence Min: 1' + '\n\n'
       for sample in self.samplelist:
           text += 'Sample: ' + str(sample) + '\n'
       kmercmdfile = os.path.join(self.cmddir,outputfolder + '_cmd.txt')
       with open(kmercmdfile,'w') as w:
           w.write(text)
       self.kmercmdfile = kmercmdfile  
       
    def calculate_kmers_cmdline_run(self,memory):
        """Runs calculate-kmers program using a command file
        
           Uses previously generated command file and subprocess module to 
           run command line. Returns information that would have been
           returned if command was run directly on command line. May be 
           preferable to run the command directly on command line to avoid 
           limitations of python. Progress can be viewed in the .log
           files in the logs folder. One log is produced for each day the 
           program is run, with the current day's log having no timestamp.
           
           WARNING: ONCE THE PROCESS IS STARTED, IT CANNOT BE TERMINATED 
           THROUGH PYTHON. THE PROCESS HAS TO BE TERMINATED USING TASK MANAGER 
           AND THE RESULTS FOLDER CREATED IN SAMPLES SHOULD BE DELETED. 
           IT MAY BE NECESSARY TO QUIT THE PYTHON KERNEL. MAKE SURE YOU HAVE
           A 64-BIT VERSION OF JAVA.
           
           Args:
               memory (int): number of gigabytes of RAM for the program to use
           Returns:
               N/A
        """
        os.chdir(self.rootdirectory)
        command = 'java -Xmx' + str(memory) + 'g'
        command += ' -jar calculate-patterns.jar ' + self.kmercmdfile
        #Get in same directory as executable
        t1 = time.time()
        #Use subprocess.run to use command line arguments
        p = subprocess.run(command,stderr=subprocess.PIPE,
                          stdout=subprocess.PIPE,universal_newlines=True,
                          shell=True)
        timing = time.time()-t1
        errors = p.stderr
        stdout = p.stdout
        self.timings.append(timing)
        self.errors.append(errors)
        self.stdout.append(stdout)    
        if errors:
            for line in errors.split('\n'):
                print(line)
            raise RuntimeError('There were errors in kmer calculation')