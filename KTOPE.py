# -*- coding: utf-8 -*-
"""
KTOPE software

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
import numpy as np
import math
from scipy import signal
import itertools
import re
import pickle
from Bio import ExPASy, SeqIO
from sklearn.cluster import KMeans 
import time 


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
    
    
class Protein:
    """Container for various characteristics of a protein
    
       Essentially acts as a struct containing protein information with trivial
       methods. Should be initialized with accession + protdirectory OR with
       organism, fullname, and sequence. If sequence isn't input directly,
       then makes sure all characters are of 20 amino acids.
       
       Attributes:
           accession (str): uniprot accession for a protein
           protdirectory (str): directory where protein pickles are found
           organism (str): full scientific name for organism
           fullname (str): verbose name for protein
           sequence (str): amino acid sequence of protein 
           valid (bool): whether protein sequence contains acceptable letters
    """
    
    def __init__(self,accession='',protdirectory='',organism='',fullname='',
                 sequence=''):
        self.accession = accession
        self.organism = organism
        self.fullname = fullname
        self.sequence = sequence
        self.protdirectory = protdirectory
        self.valid = True
        
    def __repr__(self):
        """When Protein is printed, gives name of protein"""
        return self.fullname
        
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
        
    def valid_characters(self):
        """Determines validity of sequence
        
           A sequence is considered valid if it only contains the 20 standard
           amino acids. This method sets the valid attribute of the object
           upon initialization
        """
        forbidden = ['B','J','O','U','X','Z','*','-']
        for character in forbidden:
            if character in self.sequence:
                self.valid = False
                
    def uniprot_sequence(self):
        """Uses BioPython to retrieve sequence information online
        
           Biopython has methods for retrieving sequence information using
           ExPASy which can get Uniprot information. Sets name and sequence
           attributes to values obtained online. 
           WARNING: Name may not be parsed correctly if protein has unusual
           formatting.
           
           Args:
               N/A
           Returns:
               N/A
        """
        #For searching, get rid of UniRef in sequence name
        if 'UniRef' in self.accession:
            accession = re.search('(?<=_)(.+)',self.accession).group()
        else:
            accession = self.accession
        try:
            handle = ExPASy.get_sprot_raw(accession)
        except:
            raise EnvironmentError('Uniprot search failed for ' + 
                                   self.accession)
        seq_record = SeqIO.read(handle, "swiss")
        nameregexp = '(?<=Full=)([^;]+)'
        description = seq_record.description
        self.fullname = re.findall(nameregexp,description)[0].strip()
        self.organism = seq_record.annotations['organism']
        self.sequence = str(seq_record.seq)
        self.valid_characters()
        
    def pickle_name(self):
        """Returns name of pickled protein which is just the accession"""
        picklename = self.protdirectory + os.sep + self.accession + '.pickle'
        return picklename
    
    def pickle_protein(self):
        """Pickles protein into protdirectory folder with name as accession"""
        #Get rid of characters in protein name that are problematic
        picklename = self.pickle_name()
        #For pickles, must open folder as 'wb', not 'w'
        with open(picklename,'wb') as w:
            pickle.dump(self,w)
            
    def load_protein(self):
        """Loads protein, or searches and loads protein"""
        if not self.protdirectory:
            raise ValueError('Need to input protdirectory')
        #Try to open file, if error then search online for protein
        try:
            picklename = self.pickle_name()
            with open(picklename,'rb') as f:
                protein = pickle.load(f)
            self.__dict__ = protein.__dict__
            #If it was successful
            return True
        #Protein hasn't been pickled yet
        except:
            return False
            
    def load_or_search(self):
        """Tries loading, if it doens't work, it searches the sequence"""
        loaded = self.load_protein()
        if not loaded:
            self.uniprot_sequence()
            self.pickle_protein()
             
    def add_to_seq(self,seqstring):
        """Add a short string to the existing sequence attribute"""
        #get rid of newlines or spaces
        seqstring = seqstring.strip()
        self.sequence += seqstring
    
    @classmethod    
    def parse_fasta_metadata(cls,fastaline,protdirectory=None):
        """Makes protein using metadata line from FASTA file"""
        if not fastaline.startswith('>'):
            raise ValueError('Line is not in FASTA format')
        #Special regexp for FASTA file from UniRef sequence identity clustering
        if fastaline.startswith('>UniRef'):
            accession = re.search("(?<=>)(\S+)",fastaline).group()
            fullname = re.search("(?<=\s)(.+)(?=n=)",fastaline).group().strip()
            if 'Tax=' in fastaline:
                orgexp = "(?<=Tax=)(.+)(?=TaxID)"
                organism = re.search(orgexp,fastaline).group().strip()
            #Deal with clustered sequence having no taxa listed
            else:
                shortaccession = re.search('(?<=_)(.+)',accession).group()
                #This searches Uniprot to find the organism
                tempprotein = cls(accession=shortaccession,protdirectory=
                                  protdirectory)
                tempprotein.uniprot_sequence()
                organism = tempprotein.organism
        #For normal FASTA files
        else:
            accession = re.search("(?<=\|)(.+)(?=\|)",fastaline).group()
            organism = re.search("(?<=OS=)(.+)(?=GN=)",fastaline)
            if not organism:
                organism = re.search("(?<=OS=)(.+)(?=PE=)",fastaline)
            organism = organism.group().strip()
            nameexp = "(?<=" + accession + "\|)(\w+)(.+)(?= OS=)"
            fullname = re.search(nameexp,fastaline).group(2).strip()
        protein = cls(accession,organism=organism,fullname=fullname)
        return protein
    
    @classmethod  
    def parse_fasta(cls,fastafile):
        """Reads fasta file with 1 protein and makes Protein object"""
        with open(fastafile,'r') as f:
            metadata = next(f)
            protein = Protein.parse_fasta_metadata(metadata)
            for line in f:
                protein.add_to_seq(line)
        return protein
            
                    
class Epitope:
    """Contains quantitative information about an Epitope
    
       Container for various epitope attributes that characterize an epitope.
       Attributes prevalence and centroid only have use in determining 
       consensus epitopes.Automatically calculates centroid if epitopevals has 
       valid information. Percentile, prevalence, and centroid initialized to 0
       
       Attributes:
           protein (Protein): protein from which epitope was generated
           sequence (str): The amino acid sequence of epitope
           score (float): Numerical score of epitope from AUC of frequencies
           epirange (int,int): Is range that corresponds to epitope. It's
               (inclusive, exclusive)
           epitopevals (np.array): contains score for each position in epitope 
           percentile (float): percentage of random protein epitopes that have
               scores lower than this one's score
           centroid (float): center pos of epitope in entire protein sequence
           prevalence (float): percentage of population that reacts to epitope
    """
    
    def __init__(self,protein,sequence,epirange,epitopevals):
        self.protein = protein
        self.sequence = sequence
        self.epirange = epirange
        self.epitopevals = epitopevals
        self.score = 0
        self.centroid = 0
        self.percentile = 0
        self.prevalence = 0
        #Don't calculate centroid or score if array is empty or only has 0s
        if epitopevals.size and np.count_nonzero(epitopevals):
            self.calc_centroid()
            self.score_epitope()
        
    def __repr__(self):
        """When Epitope is printed, gives various initialized attributes"""
        s = self.sequence + ' ' + str(self.score) + ' ' + str(self.epirange)
        #Only add to string representation if initialized
        if self.percentile:
            s += ' ' + str(self.percentile) + '%'
        if self.prevalence:
            s += ' ' + str(float('{:.3f}'.format(self.prevalence)))
        s += ' ' + self.protein.fullname + ' (' + self.protein.organism + ')'
        return s
        
    def __eq__(self,other):
        """Tests whether two objects have the same attributes """
        if isinstance(other, self.__class__):
            #Convert numpy arrays to bytes to allow for easy comparison
            #copy dict to avoid corrupting object by changing array type
            selfdict = dict(self.__dict__)
            otherdict = dict(other.__dict__)
            for key in selfdict.keys():
                if type(selfdict[key]) == np.ndarray:
                    selfdict[key] = np.ndarray.tobytes(selfdict[key])
                    otherdict[key] = np.ndarray.tobytes(otherdict[key])
            return selfdict == otherdict
        return NotImplemented          

    def __ne__(self,other):
        """!= is just the opposite of == """
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented
        
    def __hash__(self):
        """Object needs to have hash method to be used by set operator"""
        #Convert numpy arrays to bytes to allow for easy comparison
        #copy dict to avoid corrupting object by changing array type
        selfdict = dict(self.__dict__)
        for key in selfdict.keys():
            if type(selfdict[key]) == np.ndarray:
                selfdict[key] = np.ndarray.tobytes(selfdict[key])
        return hash(tuple(sorted(selfdict.items())))  
    
    def calc_centroid(self):
        """Calculates center of epitope in overall sequence
            
           To reduce epitope to a single value, the value at each position is
           used as a weight to find the centroid. The centroid is added to
           where the epitope is in the sequence to give a single epitope
           location in the protein sequence.
           
           Args:
               N/A
           Returns:
               N/A
        """
        weights = [i/sum(self.epitopevals) for i in self.epitopevals]
        weightedpos = []
        for i,val in enumerate(weights):
            #i+1 so first position isn't 0, to avoid multiplication by 0
            weightedpos.append((i+1)*val)
        centroidpos = np.mean(weightedpos)*len(weights)
        #Add centroid to position in protein with centroidpos indexed at 1
        self.centroid = self.epirange[0] + (centroidpos - 1)   
        
    def score_epitope(self):
        """Uses area under curve of epitope epitopevals to determine score"""
        area = np.trapz(self.epitopevals)
        score = float('{:.2f}'.format(area))
        self.score = score
        
    def percentile_calc(self,scorelist):
        """Calculates percentile for epitope given list of scores
        
          The scoreslist parameter is from randomstats.scoresamples 
          This calculation finds what the percentile of the epitope is in the
          large scorelist. Sets percentile attribute.
          
          Args:
              scorelist ([float]): list of scores from randomstats
          Returns:
              N/A
        """
        scorelen = len(scorelist)
        percentile = (np.sum(scorelist < self.score) / scorelen)
        percentile *= 100
        self.percentile = float('{:.2f}'.format(percentile))
        
class KmerInfo:
    """Stores info on kmers and loads kmers into memory
    
       Can access kmers of multiple lengths and number of defined positions.
       Also stores the enrichmin min used in kmer calculation and the 
       samples. Associates each sample with a number for easy referencing.
       Stores large enrichment dictionaries necessary for kmer tiling.
       
       Args:
           directory (str): directory in Results where kmer files are
           kmerfiles ([str]): list of filenames with kmers
           enrichmin (float): min enrichment used in kmer calculations
           samples ([Sample]): list of Sample objects in order they were used
               in kmer calculations
           sample2num ({Sample}=int): given Sample, returns its sample number
           num2sample ({int}=Sample): given sample number, returns its Sample
           enrichdicts ({int}{int}{str}{int}=float): first index is defined,
               second is positions,third index is for all the kmers,
               and fourth index is sample number. Returns the score as a float. 
               To check if something is in list use .get().
           enrichpickles ({str}=str): index is kmerfile and value is 
               corresponding pickle name for enrichdict
           kmerfiles_defpos ({str}=(int,int)): indexes (defined,positions)
               by kmerfile name
    """
    
    def __init__(self,directory,kmerfiles=None):
        self.directory = directory
        self.kmerfiles = kmerfiles
        self.enrichmin = 0
        self.samples = []
        self.sample2num = {}
        self.num2sample = {}
        self.enrichdicts = {}
        self.enrichpickles = {}
        self.kmerfiles_defpos = {}
        
    def __repr__(self):
        """When kmerFile is printed, print the directory"""
        s = 'KmerInfo directory: ' + self.directory
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
    
    def __getstate__(self):
        """Makes sure large objects are not pickled"""
        #Make deep copy to not corrupt original
        state = dict(self.__dict__)
        del state['enrichdicts']
        return state       
    
    def initialize_kmerinfo(self):
        self.get_kmerfiles()
        self.make_kmerfiles_defpos()
        self.load_metadata()
        self.sample_num_conversions()
        self.enrichdict_picklenames()

    def get_kmerfiles(self):
        """Loads in kmerfile names, does nothing if specified in init"""
        #Get filenames if files not given directly
        if self.kmerfiles:
            return
        allfiles = os.listdir(self.directory)
        #Only include files that have kmerdata
        self.kmerfiles = [i for i in allfiles if re.search(
                                                        '\ddef_\dlen.txt',i)]                       
    
    def make_kmerfiles_defpos(self):
        """Associates defined and positions numbers with each kmerfile"""
        for kmerfile in self.kmerfiles:
            defined = int(re.search('\d(?=def)',kmerfile).group())
            positions = int(re.search('\d(?=len)',kmerfile).group())
            self.kmerfiles_defpos[kmerfile] = (defined,positions)
    
    def load_metadata(self):
        """Goes through first line in kmer file and extracts parameters
        
           Uses regular expresions to figure out the Enrichment Min and samples
           from the first line of the kmerfiles.
           
           Args:
               N/A
           Returns:
               N/A
        """
        #All files have same samples and enrichmin so just open the first file
        with open(os.path.join(self.directory,self.kmerfiles[0]),'r') as f:
            firstline = next(f)
        enrichmin = re.findall('(?<=Enrichment Min: )[^;]+',firstline)[0]
        self.enrichmin = float(enrichmin)
        samplelines = re.findall('Sample: [^;|]+',firstline)
        for sampleline in samplelines:
            self.samples.append(Sample.parse_sampleline(sampleline))
            
    def sample_num_conversions(self):
        """Makes dicts that convert samples to sample numbers and vice versa"""
        for samplenum,sample in enumerate(self.samples):
            self.sample2num[sample] = samplenum
            self.num2sample[samplenum] = sample
        
    def enrichdict_picklenames(self):
        """Associates each filename with a picklename for enrichdict"""
        for kmerfile in self.kmerfiles:
            picklename,fileext = os.path.splitext(kmerfile)
            picklename = picklename + '_Enrichment_Dictionary.pickle'
            picklename = os.path.join(self.directory,picklename)
            self.enrichpickles[kmerfile] = picklename
        
    def make_enrichdicts(self):
        """Return a dictionary with score data for each sample for each kmer
        
           Acts on kmerfiles generated through kmer calculations to have
           only a single score per sample. For each kmerfile, creates 
           two-tiered dictionary where first index is kmer and second index
           is the sample number. Since samples should always be in the same
           order,this should uniquely identify the sample. The
           correct way to use an enrichdict dictionary is to use 
           enrichdict.get(kmer,{}).get(samplenum,{}). This way it will
           return an empty dictionary if kmer or samplenum isn't present.
           Pickles enrichdicts at the end.
           
           Args:
               N/A
           Returns:
               N/A
        """
        for kmerfile in self.kmerfiles:
            enrichdict = {}
            with open(os.path.join(self.directory,kmerfile),'r') as f:
                next(f)
                for line in f:
                    kmer = line.split(';')[0]
                    samplescores = line.split(';')[1:-1]
                    enrichdict[kmer] = {}
                    for samplenum,samplescore in enumerate(samplescores):
                       if samplescore:
                           enrichdict[kmer][samplenum] = float(samplescore)
            defined,positions = self.kmerfiles_defpos[kmerfile]
            if not self.enrichdicts.get(defined,False):
                self.enrichdicts[defined] = {}
            self.enrichdicts[defined][positions] = enrichdict
            picklename = self.enrichpickles[kmerfile]
            with open(picklename,'wb') as w:
                pickle.dump(enrichdict,w)
                
    def load_enrichdicts(self):
        """Loads enrichdicts into memory
        
           After calling make_enrichdicts for this dataset, this function can
           load back-in enrichdicts for initializing kmerinfo. enrichdicts
           can take a lot of RAM so be sure to monitor RAM usage and use 
           fewer kmers or samples if total RAM is exceeded.
           
           Args:
               N/A
           Returns:
               N/A
        """   
        for kmerfile in self.kmerfiles:
            picklename = self.enrichpickles[kmerfile]
            with open(picklename,'rb') as f:
                enrichdict = pickle.load(f)
            defined,positions = self.kmerfiles_defpos[kmerfile]
            if not self.enrichdicts.get(defined,False):
                self.enrichdicts[defined] = {}            
            self.enrichdicts[defined][positions] = enrichdict
 
class Tiledkmers:
    """Tiles a sequence into kmers and records their starting positions
    
        Tiling kmers for situations without wildcards is simple and uses
        a straightforward methods. The wildcard situation is more complicated
        and uses a recursive method to generates all kmers from a sequence
        based on the pairs of (defined,positions) given. Since kmers are
        added in order, also records starting positions for determining
        frequency plots. Calls kmer tiling method at initialization.
        
        Attributes:
            sequence (str): contains the sequence that kmers come from
            defpospairs ([(int,int)]): list of tuples containing 
                (defined,positions)
            seqlen (int): length of inputted sequence
            tiledkmers ({int{int}}=[str]): dict of kmers where first 
                index is # defined and second index is # positions
            startpositions ({int{int}}=[int]): dict of starting positions for
                each kmers where first index is # defined and second index 
                is # positions
    """
    
    def __init__(self,sequence,defpospairs):
        self.sequence = sequence
        self.defpospairs = defpospairs
        self.seqlen = len(sequence)
        self.tiledkmers = {}
        self.startpositions = {}
        self.tiledkmers_gen()
        
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
    
    def wildcard_gen(self,kmerlist,kmer,index,defcounter,poscounter,
                      defined,positions,startpos):
        """Recursively generates kmers for kmers with wildcards
        
           Adds one letter at a time to a kmer and outputs it to kmerlist
           once the kmer is long enough. If kmer isn't long enough, calls
           itself to add another letter or 'x' which branches the execution 
           resulting in all possible kmers being made. Only adds 'X' if the
           number of x's is less than total required (maxpositions - defined).
           Exits function if defined is reached before positions. Also adds
           startpos to startpositions when adding kmer to kmerlist.
           
           Args:
               kmerlist ([str]): list of string kmers that have been
                   found so far
               kmer (str): string with current kmer
               index (int): index in kmer based on first letter being 
                   index 0
               defcounter (int): current number of defined positions
               poscounter (int): current number of total positions
               defined (int): number of defined positions in kmer
               positions (int): the window size for a kmer
               startpos (int): position where kmer starts in sequence.
                    Indexed at 0.
            Returns:
                N/A
        """
        #If kmer is correct length, add to kmerlist
        if poscounter == positions:
            kmerlist.append(kmer)
            self.startpositions[defined][positions].append(startpos)
            return
        else:
            #If kmer has reached defined but isn't long enough, exit
            if defcounter >= defined:
                return
            xcount = poscounter - defcounter
            #Add an X if there aren't the required number yet
            if xcount < positions - defined:
                newkmer = kmer + 'x'
                self.wildcard_gen(kmerlist,newkmer,index + 1,
                                   defcounter,poscounter + 1,
                                   defined,positions,startpos)
            #Either way, add the next letter in the sequence
            newkmer = kmer + self.sequence[index + 1]
            self.wildcard_gen(kmerlist,newkmer,index + 1,
                               defcounter + 1,poscounter + 1,
                               defined,positions,startpos)
            
    def kmer_format(self,defined,positions):
        """Generates kmers from sequence using defined and positions
            
           For wildcard scenario, calls wildcard_gen for the first letter 
           that's going to be in the kmer and thus moves the window for 
           which kmers are searched. Generates kmers until there are more
           defined positions then room left in the sequence. For non-wildcard
           scenario, simply adds kmers with length positions as the starting
           position moves. Updates startpositions for non-wildcard case.
           
           Args:
               defined (int): number of defined positions in kmer
               positions (int): the window size for a kmer               
           Returns:
               kmerlist ([str]): A list of string kmers
        """
        kmerlist = []
        self.startpositions[defined][positions] = []
        for startpos in range(self.seqlen - positions + 1):
            #If no wildcards, simply tile kmers
            if defined == positions:
                kmer = self.sequence[startpos:startpos+positions]
                kmerlist.append(kmer)
                self.startpositions[defined][positions].append(startpos)
            #Use more complicated recursive wildcard method if using wildcards
            else:
                kmer = self.sequence[startpos]
                self.wildcard_gen(kmerlist,kmer,startpos,1,1,
                                   defined,positions,startpos)
        return kmerlist
        
    def tiledkmers_gen(self):
        """Tiles sequence into kmers by calling other methods
        
           Loops through all combinations of defined and positions based on
           initialization parameters and calls kmer_format for each combo.
           kmer_format then calls wildcard_gen for each window in the 
           sequence in wildcard case. Alters tiledkmer attribute of object 
           where the first index is defined positions and second one is number
           of positions.
           
           Args:
               N/A
           Returns:
               N/A
        """
        for pair in self.defpospairs:
            defined = pair[0]
            positions = pair[1]
            #Initialize tiledkmers dictionary if needed
            if not self.tiledkmers.get(defined,False):
                self.tiledkmers[defined] = {}
                #Also initialize startpositions
                self.startpositions[defined] = {}
            kmerlist = self.kmer_format(defined,positions)
            self.tiledkmers[defined][positions] = kmerlist
                              
class FrequencyInfo:
    """Contains frequency information after aligning kmers to sequence
    
       kmers are aligned to the sequence and then the alignment is used
       to score each position and ultimately generate epitopes. Contains
       separate functions to add kmers for individual or group analysis.
       
       Attributes:
           protein (Protein): protein from which sequence comes from
           spanval (int): window size to average the frequency for each 
               position. Must be odd
           edgeweight (int): value that end positons in window are weighted.
               Positions closer to center are linearly weighted with center
               value being weighted as 1
           poscounters (np.array[float]): contains frequency information for
               each position in sequence
           smoothedcounters (np.array[float]): contains frequency information
               after using spanned_freq
    """
    
    def __init__(self,protein,spanval,edgeweight):
        self.protein = protein
        self.spanval = spanval
        self.edgeweight = edgeweight
        self.poscounters = np.zeros(len(self.protein.sequence))
        self.smoothedcounters = np.zeros(len(self.protein.sequence))
        
    def __eq__(self,other):
        """Tests whether two objects have the same attributes """
        if isinstance(other, self.__class__):
            #Convert numpy arrays to bytes to allow for easy comparison
            #copy dict to avoid corrupting object by changing array type
            selfdict = dict(self.__dict__)
            otherdict = dict(other.__dict__)
            for key in selfdict.keys():
                if type(selfdict[key]) == np.ndarray:
                    selfdict[key] = np.ndarray.tobytes(selfdict[key])
                    otherdict[key] = np.ndarray.tobytes(otherdict[key])
            return selfdict == otherdict
        return NotImplemented          

    def __ne__(self,other):
        """!= is just the opposite of == """
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented
        
    def __hash__(self):
        """Object needs to have hash method to be used by set operator"""
        #Convert numpy arrays to bytes to allow for easy comparison
        #copy dict to avoid corrupting object by changing array type
        selfdict = dict(self.__dict__)
        for key in selfdict.keys():
            if type(selfdict[key]) == np.ndarray:
                selfdict[key] = np.ndarray.tobytes(selfdict[key])
        return hash(tuple(sorted(selfdict.items())))       
                 
    def add_kmer(self,kmer,startpos,score):
        """Takes in a kmer and adds it to instance for individual scoring
        
           Using the begin position of the kmer in the sequence, adds 
           score to each position contained in the kmer that isn't 'x'.
           Uses log base 2 of enrichment value.
           
           Args:
               kmer (str): kmer to be added to Frequencies
               startpos (int): position where kmer begins in sequence
               score (float): value of score for the kmer
           Returns:
               N/A
        """
        for i in range(len(kmer)):
            if kmer[i] != 'x':
                incrementval = math.log2(score)
                self.poscounters[startpos + i] += incrementval                 
        
    def smoothed_frequency(self):
        """Smooths data by averaging over intervals
            
           For span of n, the left and right (n-1)/2 positions are included
           and a weighted average is taken for the value at the middle position
           where farther away positions are weighted less. If a position has
           a frequency of 0 in poscounters, it retains a frequency of 0 in 
           smoothedpositions so that epitopes stay distinct. Smoothed frequency
           data aids with finding epitopes.
           
           Args:
               N/A
           Returns:
               N/A
        """
        #The sequence needs to be at least as long as the span and must be odd
        if len(self.poscounters) < self.spanval or not self.spanval % 2:
            raise ValueError('spanval is too long or is not an odd number')
        #Span of 1 doesn't alter frequencies
        if self.spanval == 1:
            self.smoothedcounters = self.poscounters
        sidespan = self.spanval // 2
        sideweights = np.linspace(self.edgeweight,1,sidespan+1)
        #Combine the side weights and the reversed sideweights, without
        #duplicating the 1 in the middle
        weights = np.concatenate([sideweights,sideweights[::-1][1:]])
        #iterate through each position that can exist with given span
        for i in range(sidespan,len(self.poscounters) - sidespan):
            #make array of relevant values to left and right of desired value
            spanvals = self.poscounters[i-sidespan:i+sidespan+1]
            newval = np.average(spanvals,weights=weights)
            #If a position is 0 in poscounters, it is 0 in smoothedcounters
            if not self.poscounters[i]:
                newval = 0
            self.smoothedcounters[i] = newval
                                 
    def epitope_evaluate(self,sequence,startpos):
        """Scores epitope using different frequencyinfo than originally
        
           Calculates area under the curve for a given subsequence (epitope) in
           the whole protein sequence using the starting position. Intended for
           an epitope found using a first sample, and then scored using 
           poscounters from a second sample.
        """
        endpos = startpos + len(sequence)
        epirange = (startpos,endpos)
        epitopevals = self.poscounters[slice(*epirange)]
        epitope = Epitope(self.protein,sequence,epirange,epitopevals)
        return epitope
                                     
    
class ProteinEpitopes:
    """Finds all the epitopes in a protein sequence for samples
    
       Tiles protein sequence into kmers, then populates FrequencyInfo
       objects using individual or group scoring. Each sample has a separate
       FrequencyInfo object. Finds epitopes for each sample. Finds consensus
       epitopes for a population that meet a prevalence threshold
       
       Attributes:
           protein (Protein): protein for which epitopes are being ofund
           kmerinfo (KmerInfo): contains enrichment info for kmers
           specificsamples ([Sample]): list of samples to be analyzed. If not
               specificed, all samples are used
           epiminlen (int): the minimum length of an epitope
           epimaxlen (int): the maximum length of an epitope
           nameaddendum (str): suffix of the randomstats file 
           randomstats (RandomStats): randomstats file corresponding to data.
               Will be loaded automatically if not specified but for examining
               multiple proteins, its better to only load it once.
           spanval (int): window size to average the frequency for each 
               position. Must be odd
           edgeweight (int): value that end positons in window are weighted.
               Positions closer to center are linearly weighted with center
               value being weighted as 1                
           samplenums = ([int]): numbers corresponding to samples used
           defpospairs ([(int,int)]): list of tuples of (defined,positions) 
               used in calculations
           tiledkmers (Tiledkmers): contains kmers tiled from protein
               sequence and where kmers start
           frequencydict ({int}=FrequencyInfo): FrequencyInfo objects for 
               each sample, indexed by samplenum
           epitopedict ({int}=[Epitope]): lists of epitopes found for each 
               sample, indexed by samplenum
           consensusepitopes ([Epitope]): list of epitopes found using 
               consensus_epitopes, prevalent and meeting epitopethresh
           epitopescores = ({Epitope}{int}={str:float,str:float}):
               scores and percentiles indexed by epitope and samplenum.
               Keys are 'score' and 'percentile'
    """

    def __init__(self,protein,kmerinfo,specificsamples=None,epiminlen=6,
                 epimaxlen=15,nameaddendum='',randomstats=None,spanval=7,
                 edgeweight=0.1):
        self.protein = protein
        self.kmerinfo = kmerinfo
        self.specificsamples = specificsamples
        #Default to analyzing all samples
        if not self.specificsamples:
            self.specificsamples = self.kmerinfo.samples
        self.epiminlen = epiminlen
        self.epimaxlen = epimaxlen
        self.nameaddendum = nameaddendum
        self.randomstats = randomstats
        self.spanval = spanval
        self.edgeweight = edgeweight        
        self.samplenums = []
        self.defpospairs = []
        self.tiledkmers = None
        self.frequencydict = {}
        self.epitopedict = {}
        self.consensusepitopes = None
        self.epitopescores = {}
        
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
    
    def __getstate__(self):
        """Makes sure large objects are not pickled"""
        #Make deep copy to not corrupt original
        state = dict(self.__dict__)
        del state['kmerinfo']
        del state['randomstats']
        return state    

    def initialize_proteinepitopes(self,calcstatistics=True):
        """Initializes proteinepitopes by running functions in correct order"""
        self.get_samplenums()
        self.initialize_tiledkmers()
        self.calculate_frequencyinfo()
        self.find_epitopes() 
        if calcstatistics:
            if not self.randomstats:
                self.randomstats = RandomStats.load_randomstats(
                                            self.kmerinfo,self.nameaddendum)
            self.epitope_statistics()

    def get_samplenums(self):
        """Translates sample names into samplenums for indexing"""
        for sample in self.specificsamples:
            samplenum = self.kmerinfo.sample2num[sample]
            self.samplenums.append(samplenum)
            
    def initialize_tiledkmers(self):
        """Initializes Tilekmers object using kmerinfo"""
        for kmerfile in self.kmerinfo.kmerfiles:
            defined,positions = self.kmerinfo.kmerfiles_defpos[
                                                                   kmerfile]
            self.defpospairs.append((defined,positions))
        tiledkmers = Tiledkmers(self.protein.sequence,self.defpospairs)
        self.tiledkmers = tiledkmers
        
    def calculate_frequencyinfo(self):
        """Calculates FrequencyInfo objects for each sample
        
           Goes through each sample and adds kmers contained in 
           tiledkmers poscounters object of FrequencyInfo. Also calculates
           smoothedcounters which is used for epitope finding, but not for 
           scoring.
           
           Args:
               N/A
           Returns:
               N/A
        """
        for samplenum in self.samplenums:
            frequencyinfo = FrequencyInfo(self.protein,self.spanval,
                                          self.edgeweight)
            for pair in self.defpospairs:
                defined,positions = pair[0],pair[1]
                kmers = self.tiledkmers.tiledkmers[defined][positions]
                startpos = self.tiledkmers.startpositions[defined][
                                                                    positions]
                enrichdict = self.kmerinfo.enrichdicts[defined][positions]
                for kmer,start in zip(kmers,startpos):
                    score = enrichdict.get(kmer,{}).get(samplenum,{})
                    #kmer makes no contribution if it doesn't have a score
                    if score:
                        frequencyinfo.add_kmer(kmer,start,score)
            frequencyinfo.smoothed_frequency()
            self.frequencydict[samplenum] = frequencyinfo
                              
    def eliminate_consecutive(self,numlist):
        """Eliminates consecutive numbers in a list
        
           When using signal.argelextrema, any zeros that follow eachother with
           the >= or <= will be included, but only the first one is really an
           extreme. Therefore, it's necessary to eliminate consecutive 
           positions that are extrema, since they're usually zeros. groupby
           looks at the difference between the extremum position and it's index
           in the list of extrema. When this difference changes, that means
           there is a non-consecutive value and a new group is formed. group[0]
           is the first member of the group and group[0][1] is the extremum
           position of the first member of the group
           
           Args:
               numlist (list[int]): contains list to have consecutives removed
           Returns:
               listfinal (list[int]): processed list without consecutives
        """
        listfinal = []
        for key, group in itertools.groupby(enumerate(numlist), 
            lambda item: item[0] - item[1]):
            listfinal.append(list(group)[0][1])  
        return listfinal  
    
    def epitope_finding_edgecases(self,smoothedcounters,maxima,minima):
        """Deals with edge cases that occur with finding maxima and minima
        
           There are a number of issues that can occur when finding maxima
           and minima in this data, most commonly, zeros in the wrong places.
           
           Args:
               smoothedcounters (np.array): array of frequency values relating
                   to FrequencyInfo.smoothedcounters
               maxima ([int]): positions where maxima were found in 
                   smoothedcounters
               minima ([int]): positions where minima were found in 
                   smoothedcounters
            Returns:
                maxima ([int]): maxima list with edge cases fixed
                minima ([int]): minima list with edge cases fixed
        """
        #account for the difference when maxima or minima is first
        if minima[0] != 0 and maxima[0] < minima[0]:
            minima = np.insert(minima,0,0)
        #If last position is a maximum, add a minimum after it
        if maxima[-1] > minima[-1]:
            minima = np.append(minima,len(smoothedcounters) - 1)
        #0 can't be a maximum and a minimum. Make it a minimum in this case
        if maxima[0] == 0 and minima[0] == 0:
            maxima = np.delete(maxima,0)
        return maxima,minima
    
    def find_epitopes(self):
        """Generates lists of epitopes from poscounters data for each sample
        
           Extracts intervals from FrequencyInfo which contain maaxima,
           since these may be potential epitopes. It finds all maxima and
           minima, treats the data for edge cases, and then makes an epitope 
           between two minima if a maxima occurs between them. These epitope
           intervals are then shortened to account for maxepilen and to
           exclude zeros. Epitopes must be of sufficient length to be stored.
           Each interval, it's AUC score, its values, and it's 
           sequence are made into an Epitope object and added to a list. 
           
           Args:
               N/A
           Returns:
               N/A
        """
        #Loop through each sample's frequencyinfo object
        for samplenum in self.frequencydict.keys():
            epitopes = []
            frequencyinfo = self.frequencydict[samplenum]
            smoothedcounters = frequencyinfo.smoothedcounters
            poscounters = frequencyinfo.poscounters
            minima = signal.argrelextrema(np.array(smoothedcounters),
                                                np.less_equal)[0]
            maxima = signal.argrelextrema(np.array(smoothedcounters),
                                                np.greater_equal)[0]
            #Get rid of zeros in maxima due to >= maxima finding
            maxima = [i for i in maxima if smoothedcounters[i] != 0]
            #Add empty list and go to next sample
            if len(maxima) == 0 or len(minima) == 0:
                self.epitopedict[samplenum] = epitopes
                continue
            #Need to get rid of consecutive zeros in minima and then maxima
            minima = self.eliminate_consecutive(minima)
            maxima = self.eliminate_consecutive(maxima)
            #Deal with all edge cases
            maxima,minima = self.epitope_finding_edgecases(smoothedcounters,
                                                           maxima,minima)
            #Make intervals out of minima that become epitopes if contain max
            for index in range(len(minima)-1):
                epirange = (minima[index],minima[index+1])
                #get rid of positions in range that are 0
                nonzeropos = [i for i in range(*epirange) if 
                              smoothedcounters[i] != 0]
                epirange = (nonzeropos[0],nonzeropos[-1])
                epitopelen = epirange[1] - epirange[0]
                #epirange needs to have at least one maximum contained in it
                found = False
                for maximum in maxima:
                    if maximum >= epirange[0] and maximum <= epirange[1]:
                        found = True
                        #Stop looking for maxima once one is found
                        break
                #only store epitope if it is long enough
                if found and epitopelen >= self.epiminlen:
                    #Trim ends of epitope to get to maxlength or 1 less
                    if epitopelen > self.epimaxlen:
                        sidepos = int(np.ceil((epitopelen-self.epimaxlen)/2))
                        epirange = (epirange[0]+sidepos,epirange[1]-sidepos)
                    epitopevals = poscounters[slice(*epirange)]
                    sequence = self.protein.sequence[slice(*epirange)]
                    #make Epitope object out of information
                    epitope = Epitope(self.protein,sequence,epirange,
                                      epitopevals)
                    epitopes.append(epitope)
            #Sort epitopes by score
            epitopes = sorted(epitopes,key=lambda epi: epi.score,reverse=True)
            self.epitopedict[samplenum] = epitopes            
                            
    def epitope_statistics(self):
        """Use RandomStats to assign percentile to each epitope"""
        for samplenum in self.samplenums:
            for epitope in self.epitopedict[samplenum]:
                epilen = len(epitope.sequence)
                scorelist = self.randomstats.samplescores[samplenum][epilen]
                epitope.percentile_calc(scorelist)
            epitopelist = self.epitopedict[samplenum]
            sortedepitopes = sorted(epitopelist,key=lambda epi: epi.percentile,
                                    reverse=True)
            self.epitopedict[samplenum] = sortedepitopes
                            
    def group_scores(self,epitope):
        """Determines scores for an epitope corresponding to the protein
           
           The sequence and startpos of an epitope are used to determine its
           score in every sample used in this object's samplenums. Useful for
           consensus epitopes.
           
           Args:
               epitope (Epitope): epitope to be evaluated in all samples
           Returns:
               epitopescores ({int}={str:float,str:float}): gives score and 
                 percentile for each sample for the epitope. keys are 'score'
                 and 'percentile'
        """    
        epitopescores = {}
        for samplenum in self.samplenums:
            freqinfo = self.frequencydict[samplenum]
            newepitope = freqinfo.epitope_evaluate(epitope.sequence,
                                                   epitope.epirange[0])
            epitopelen = len(newepitope.sequence)
            scorelist = self.randomstats.samplescores[samplenum][epitopelen]
            newepitope.percentile_calc(scorelist)
            scoreinfo = {'score':newepitope.score,
                         'percentile':newepitope.percentile}
            epitopescores[samplenum] = scoreinfo
        return epitopescores                            
                     
    def compile_epitopes(self,scorecutoffs):
        """Compiles all epitopes from all samples that meet a cutoff
           
           All samples are added to the allepitopes list if they meet a 
           statistical threshold.
           
           Args:
               scorecutoffs ({int}{int}=[float]): scores cutoff for statisical
                   significance. First index is samplenum, second is length of
                   epitope
           Returns:
               allepitopes ([Epitope]): list of all epitopes found for all
                   samples.
        """
        allepitopes = []
        #Make list of all epitopes
        for samplenum in self.samplenums:
            for epitope in self.epitopedict[samplenum]:
                epitopelen = len(epitope.sequence)
                #Only cluster significant epitopes and make sure epitopes exist
                if epitope and (epitope.score >= 
                                scorecutoffs[samplenum][epitopelen]):
                    allepitopes.append(epitope)
        return allepitopes
    
    def kmeans_clustering_greedy(self,centroids,allepitopes,randomseed,n_init):
        """Clusters are formed from epitope centroids to group similar epitopes
        
           By clustering epitopes, similar epitopes in different samples can
           be associated with each other. This allows for epitopes that
           slightly overlap with each other to be found. Starts with 2 clusters
           and increases cluster number until the cluster center is found 
           in the sequence of each epitope in the cluster. Tends to make empty
           clusters which are removed. Automatically determines # of threads.
           
           Args:
               centroids ([float]): centroids of allepitopes
               allepitopes ([Epitope]): list of all epitopes found for all
                   samples.
               randomseed (int): seed used to run KMeans algorithm
               n_int (int): number of iterations for each call to Kmeans
           Returns:
               groups ([[Epitope]]): List of groups of epitopes that are 
                   clustered together
               
        """
        finished = False
        clusters = 2
        while not finished:
            kmeans = KMeans(n_clusters=clusters,random_state=randomseed,
                            n_init=n_init)
            kmeans.fit(centroids)
            clustercenters = kmeans.cluster_centers_
            groups = [[] for i in range(clusters)]
            for cluster in range(clusters):
                for i in range(len(allepitopes)):
                    if kmeans.labels_[i] == cluster:
                        groups[cluster].append(allepitopes[i])
            finished = self.check_clustergroups(groups,clustercenters)
            clusters += 1
        #Get rid of empty groups
        print(clusters)
        groups = [i for i in groups if i]
        return groups 
    
    def start_clustering(self,centroids,allepitopes,randomseed):
        """Clusters are formed from epitope centroids to group similar epitopes
        
           By clustering epitopes, similar epitopes in different samples can
           be associated with each other. This allows for epitopes that
           slightly overlap with each other to be found.Tends to make empty
           clusters which are removed. Automatically determines # of threads.
           Starts clustering by using binary division scheme where upper and
           lower limits on cluster number are determined and then a recursive
           function finds the optimal cluster number.
           
           Args:
               centroids ([float]): centroids of allepitopes
               allepitopes ([Epitope]): list of all epitopes found for all
                   samples.
               randomseed (int): seed used to run KMeans algorithm
           Returns:
               groups ([[Epitope]]): List of groups of epitopes that are 
                   clustered together
               
        """
        lower = 2
        upper = 2
        foundupper = False
        #Determine initial upper and lower bounds for cluster number
        while not foundupper:
            upper *= 2
            groups,lowerstatus = self.kmeans_clustering(lower,centroids,
                                                        allepitopes,randomseed)
            #If more clusters than centroids set upper to centroid number
            if upper >= len(centroids):
                upper = len(centroids)
                #If upper is odd, make it even to avoid fractional index
                #upper is never used to cluster again so it won't give error
                if upper % 2:
                    upper += 1
                foundupper = True
            else:
                groups,upperstatus = self.kmeans_clustering(upper,centroids,
                                                     allepitopes,randomseed)
            #Don't enter if foundupper is true from earlier loop
            #foundupper=true when lower is False and upper is True
            if not foundupper and not lowerstatus and upperstatus:
                foundupper = True
            #Only set lower to upper if going through another iteration
            if not foundupper:
                lower = upper
        #Find optimal number of clusters and return groups
        groups = self.recursive_clustering(lower,upper,centroids,allepitopes,
                                           randomseed)
        return groups
        
    
    def kmeans_clustering(self,clusters,centroids,allepitopes,randomseed):
        """Uses SciKit's KMeans function to cluster centroids
        
           Given centroids, epitopes and number of clusters, clusters centroids
           and therefore epitopes into groups. Empty groups are removed.
           Also determines whether clusters are good enough.
           
           Args:
               clusters (int): number of clusters for KMeans
               centroids ([float]): centroids of allepitopes
               allepitopes ([Epitope]): list of all epitopes found for all
                   samples
               randomseed (int): seed used to run KMeans algorithm 
           Returns:
               groups ([[Epitope]]): List of groups of epitopes that are 
                   clustered together
               finishedstatus (bool): whether all cluster means are contained 
                   in epitopes in cluster   
        """
        kmeans = KMeans(n_clusters=clusters,random_state=randomseed)
        kmeans.fit(centroids)
        clustercenters = kmeans.cluster_centers_
        groups = [[] for i in range(clusters)]
        for cluster in range(clusters):
            for i in range(len(allepitopes)):
                if kmeans.labels_[i] == cluster:
                    groups[cluster].append(allepitopes[i])
        finishedstatus = self.check_clustergroups(groups,clustercenters)
        #Remove empty groups
        groups = [i for i in groups if i]
        return groups,finishedstatus
    
    def recursive_clustering(self,lower,upper,centroids,allepitopes,
                             randomseed):
        """Recursive method that finds optimal number of clusters and returns
        
           Uses upper and lower limit to evaluate middle of cluster in binary
           division. Calls itself to either use upper half of division or
           lower half of division. Finishes when it finds changeover from
           check_clustergroups returning False to True. If False and True
           oscillate a little, may not pick up first instance of True.
           
           Args:
               lower (int): lower number of clusters
               upper (int): upper number of clusters
               centroids ([float]): centroids of allepitopes
               allepitopes ([Epitope]): list of all epitopes found for all
                   samples
               randomseed (int): seed used to run KMeans algorithm 
           Returns:
               groups ([[Epitope]]): List of groups of epitopes that are 
                   clustered together
        """
        #Use ceiling function to ensure last clustering is 3 consecutive nums 
        middle = math.ceil((lower + upper) / 2)
        middlelowerdiff = middle - lower
        groups,middlestatus = self.kmeans_clustering(middle,centroids,
                                                     allepitopes,randomseed)
        if middlelowerdiff == 1:
            if middlestatus:
                return groups
            else:
                #If middle is false, then re-calculate upper groups and return
                groups,upperstatus = self.kmeans_clustering(upper,centroids,
                                                            allepitopes,
                                                            randomseed)
                return groups
        if middlestatus:
            return self.recursive_clustering(lower,middle,centroids,
                                             allepitopes,randomseed)
        else:
            return self.recursive_clustering(middle,upper,centroids,
                                             allepitopes,randomseed)
            
    def check_clustergroups(self,groups,clustercenters):
        """Checks whether each cluster center is in epirange of each epitope"""
        for i,group in enumerate(groups):
            center = clustercenters[i]
            for epitope in group:
                if (epitope.epirange[0] < center and 
                    epitope.epirange[1] > center):
                    continue
                else:
                    return False
        return True
        
    def cluster_representative(self,group):
        """Takes a cluster and make a new epirange of a representative epitope
        
           First it determines what the maximum epirange could be, based on the
           first and last positions appearing in each epitope. Determines the
           length of the representative to be the average length of epitopes in
           the cluster. Determines the highest scoring positons in a combined
           vector of all scores at each position. Trims ends based on which
           end has a lower score to meet epitopelength and outputs epirange.
           
           Args:
               group (list[Epitope]): list of epitopes outputted by kmeans
           Returns:
               newepirange ((int,int)): epirange corresponding to the
                   representative epitope of the cluster.
        """
        #Determine maximum length of epitope
        epirange = [100000,-1]
        for epitope in group:
            if epitope.epirange[0] < epirange[0]:
                epirange[0] = epitope.epirange[0]
            if epitope.epirange[1] > epirange[1]:
                epirange[1] = epitope.epirange[1]
        epitopelength = int(np.average([len(i.sequence) for i in group]))
        scorevector = [[] for i in range(*epirange)]
        for epitope in group:
            begin = epitope.epirange[0] - epirange[0]
            for j,score in enumerate(epitope.epitopevals):
                scorevector[begin + j].append(score)
        scorevector = [np.mean(i) for i in scorevector]
        posdict = {i:scorevector[i] for i in range(len(scorevector))}
        toppos = sorted(posdict, key=posdict.get,reverse=True)[:epitopelength]
        start,end = min(toppos),max(toppos)
        currentlen = end - start + 1
        #Trim edges if too long
        while currentlen > epitopelength:
            tempvector = scorevector[start:end+1]
            if tempvector[0] >= tempvector[-1]:
                end -= 1
            else:
                start += 1
            currentlen = end - start + 1
        newepirange = (epirange[0] + start,epirange[0] + end + 1)
        return newepirange
    
    def remove_epitope_redundancy(self,epitopes):
        """Removes all but one of epitopes with the same sequences
        
           Cycles through epitopes and keeps only one copy of an 
           epitopes that have same sequence. 
           
           Args:
               epitopes ([Epitope]): epitope list with redundancy
           returns:
               nonredundant ([Epitope]): nonredundant epitope list
        """
        nonredundantlist = []
        for epitope in epitopes:
            found = False
            for nonredundant in nonredundantlist:
                if epitope.sequence == nonredundant.sequence:
                    found = True
            if not found:
                nonredundantlist.append(epitope)
        return nonredundantlist           
        
    def remove_close_clusters(self,consensusepitopes,clusterdistance):
        """Curates representative epitopes by removing ones that are too close
        
           Clustering isn't perfect and for some highly antigenic regions,
           multiple clusters might be found for the same epitope. This removes
           epitopes whose centroids are within a certain distance of eachother.
           Lower prevalence epitopes are removed.
           
           Args:
               consensusepitopes ([Epitope]): epitopes found in a high enough
                   portion of the population
               clusterdistance (int): minimum distance epitope centroids must
                   be from each other
           Returns:
               keptepitopes ([Epitope]): epitope list with close epitopes
                   removed
        """
        keptepitopes = []
        for epitope in consensusepitopes:
            found = False
            for keptepitope in keptepitopes:
                if (abs(epitope.centroid - keptepitope.centroid) <=
                    clusterdistance):
                    found = True
            if not found:
                keptepitopes.append(epitope)
        return keptepitopes        
    
    def add_epitopescores(self,consensusepitopes):
        """Adds scores for epitopes to self.epitopescores"""
        for consensusepitope in consensusepitopes:
            consensusscores = self.group_scores(consensusepitope)
            self.epitopescores[consensusepitope] = consensusscores
    
    def consensus_epitopes(self,epitopethresh,minprev,randomseed=0,
                           clusterdistance=10):
        """Uses Kmeans to cluster epitopes and find which one are prevalent
        
           Outputs a list of epitopes that are nonredundant and meet the 
           epitopethresh and minprev thresholds. Adds scores of epitopes to
           epitopescores attribute. Consists of following steps:
           
           1. Add all epitopes from all samples to a list if they meet
               epitopethresh
           2. Make a list of all centroids from all epitopes
           3. Cluster centroids using kmeans with increasing values of K until
               for each cluster, the center exists in each epitope's epirange
           4. For each cluster, make a representative epitope and keep it if
               it meets epitopethresh in minprev fraction of samples
           5. Sort representative epitopes by prevalence and remove redundancy
           
           Args:
               epitopethresh (int): percentile cutoff for an epitope to be
                  significant
               minprev (float): minimum fraction of samples epitope needs to
                   be in 
               randomseed (int): seed for rng for kmeans
               clusterdistance (int): minimum distance epitope centroids must
                   be from each other
               njobs (int): number of iterations in each K-means iteration  
           Returns:
               N/A
        """
        #Round minimum prevalence down to be conservative, requires ># samples
        minprevsamples = math.ceil(minprev*len(self.samplenums))
        self.randomstats.scores_for_percentile(epitopethresh)
        scorecutoffs = self.randomstats.scorecutoffs        
        allepitopes = self.compile_epitopes(scorecutoffs)
        centroids = []
        for epitope in allepitopes:
            centroids.append([epitope.centroid])            
        centroids = np.array(centroids)
        #If there are no epitopes, set consensus to empty and exit function
        if not centroids.size:
            self.consensusepitopes = []
            return
        #If 1 epitopes, just make epitope the group    
        elif len(centroids) == 1:
            groups = [allepitopes]
        #For 2 epitope case, first see if 1 cluster works
        elif len(centroids) == 2:
            #If 1 cluster works, use group with 2 epitopes in it
            groups,finishedstatus = self.kmeans_clustering(1,centroids,
                                                     allepitopes,randomseed)
            #If 1 cluster doesn't work, then put epitopes in 2 groups
            if not finishedstatus:
                groups = [[allepitopes[0]],[allepitopes[1]]]
        #If more than 2 centroids, use normal clustering function
        else:
            groups = self.start_clustering(centroids,allepitopes,randomseed)
        consensusepitopes = []
        for group in groups:
            epirange = self.cluster_representative(group)
            sequence = self.protein.sequence[slice(*epirange)]              
            numpresent = 0
            valuevector = [[] for i in range(*epirange)]
            for count,samplenum in enumerate(self.samplenums):
                samplevals = self.frequencydict[samplenum].poscounters[
                                                            slice(*epirange)]
                sampleepitope = Epitope(self.protein,sequence,epirange,
                                        samplevals)
                epitopelen = len(sampleepitope.sequence)
                if sampleepitope.score >= scorecutoffs[samplenum][epitopelen]:
                    numpresent += 1
                    #Only add prevalent epitopes to values
                    for j,score in enumerate(sampleepitope.epitopevals):
                        valuevector[j].append(score)
            if numpresent >= minprevsamples:                    
                epitopevals = np.array([np.mean(i) for i in valuevector if i])        
                consensusepitope = Epitope(self.protein,sequence,epirange,
                                       epitopevals)
                #Set prevalence for epitope
                prevalence = numpresent/len(self.samplenums)
                consensusepitope.prevalence = float('{:.3f}'.format(
                                                                  prevalence))
                consensusepitopes.append(consensusepitope)     
        #Sort and remove redundancy and close epitopes from list
        #Sort consensusepitopes by score than prevalence, so highest on list
        #have highest prevalence, and same prevalence are ranked by score
        consensusepitopes = sorted(consensusepitopes,key=lambda i: 
                                   (i.prevalence,i.score),reverse=True)     
        consensusepitopes = self.remove_epitope_redundancy(consensusepitopes)
        consensusepitopes = self.remove_close_clusters(consensusepitopes,
                                                       clusterdistance)
        self.consensusepitopes = consensusepitopes
        #Add scores to epitopescores dict 
        self.add_epitopescores(consensusepitopes)
    
    
class RandomStats:
    """Calculates epitope scores for random proteins to generate statistics
    
       To know whether a particular score for an epitope is significant, it 
       needs to be compared to "typical" scores. This class takes a number of 
       random proteins and finds epitopes and scores them. This allows a
       percentile to be attached to an epitope which indicates its signficance.
       Should be run with same epiminlen,epimaxlen,spanval, and edgeweights
       as ProteinEpitopes.Finds statistics for all samples used in KmerInfo.
       
       Attributes:
           kmerinfo ([KmerInfo]): kmer data used for calculations
           numproteins (int): number of proteins contained and scored
           allproteinpickle (str): filename of pickled list of all proteins
           epiminlen (int): min length for epitopes
           epimaxlen (int): max length for epitopes
           seqmin (int): minimum length of sequences used. Should be equal
               to at least 2 times the number of maximum positions. If sequence
               is too short, there is no way to tile it.  
           randomseed (int): seed for random number generator   
           nameaddendum (str): suffix of the randomstats file in case multiple 
               randomstats files are generated for the same directory                   
           spanval (int): window size to average the frequency for each 
               position. Must be odd
           edgeweight (int): value that end positons in window are weighted.
               Positions closer to center are linearly weighted with center
               value being weighted as 1  
           samples ([Sample]): list of samples in kmerinfo
           proteins ([Protein]): proteins used in calculations
           samplescores ({int}{int}=[float]): scores for epitopes indexed by 
                sample numbers then epitope length
           scorecutoffs ({int}{int}=[float]): cutoff score corresponding to a
               percentile in samplescores. Indexed by sample numbers then 
               epitope length
           picklename (str): name for pickled version of instance
    """
    
    def __init__(self,kmerinfo,numproteins,allproteinpickle='',epiminlen=6,
                 epimaxlen=15,seqmin=10,randomseed=0,nameaddendum='',spanval=7,
                 edgeweight=0.1):
        self.kmerinfo = kmerinfo
        self.numproteins = numproteins
        self.allproteinpickle = allproteinpickle        
        self.epiminlen = epiminlen
        self.epimaxlen = epimaxlen        
        self.seqmin = seqmin
        self.randomseed = randomseed
        self.nameaddendum = nameaddendum
        self.spanval = spanval
        self.edgeweight = edgeweight
        self.samples = self.kmerinfo.samples
        self.proteins = []
        self.samplescores = {}
        self.scorecutoffs = {}
        self.picklename = ''
        self.picklename = RandomStats.pickle_name(self.kmerinfo,
                                                  self.nameaddendum)
        np.random.seed(self.randomseed)
        
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

    def __getstate__(self):
        """Makes sure large objects are not pickled"""
        #Make deep copy to not corrupt original
        state = dict(self.__dict__)
        del state['kmerinfo']
        return state
    
    @staticmethod
    def pickle_name(kmerinfo,nameaddendum=''):
        """Make filename for RandomStats object"""
        resultsfolder,result = os.path.split(kmerinfo.directory)
        picklename = os.path.join(kmerinfo.directory,result)
        picklename += '_RandomStats' + nameaddendum + '.pickle'
        return picklename
    
    @staticmethod
    def load_randomstats(kmerinfo,nameaddendum=''):
        """Loads Randomstats object from pickle using kmerinfo"""
        picklename = RandomStats.pickle_name(kmerinfo,nameaddendum)
        with open(picklename,'rb') as f:
                randomstats = pickle.load(f)
        return randomstats
        
    def make_all_proteins(self,allproteinfile):
        """Reads all proteins in a fastafile into memory
        
           Intended to be used on all reviewed proteins from uniprot which is
           on the order of 500,000 proteins so it takes up a large amount of
           memory. Proteins for randomstats calculations will be picked from
           this list. Only needs to be called once. Doesn't pickle proteins 
           that it searches.
           
           Args:
               allproteinfile (str): FASTA file containing all reviewed 
                   proteins
           Returns:
               N/A
        """
        allproteins = []
        currentprot = ''
        with open(allproteinfile,'r') as f:
            for line in f:
                if line.startswith('>'):
                    if currentprot != '':
                        allproteins.append(currentprot)
                    newprotein = Protein.parse_fasta_metadata(line)
                    currentprot = newprotein
                else:
                    currentprot.add_to_seq(line)
            #Add last protein         
            allproteins.append(currentprot)
        for protein in allproteins:
            protein.valid_characters()
        #Save in same directory as allproteinfile
        directory = os.path.dirname(allproteinfile)
        allproteinpickle = os.path.join(directory,'all_proteins.pickle')
        self.allproteinpickle = allproteinpickle
        with open(allproteinpickle,'wb') as w:
            pickle.dump(allproteins,w)    
            
    def load_all_proteins(self):
        """Loads file created with make_all_proteins"""
        with open(self.allproteinpickle,'rb') as f:
            allproteins = pickle.load(f)
        return allproteins    
    
    def random_proteins(self):
        """Generates a list of random proteins 
        
           Picks a random numproteins quantity of sequences from  
           allproteinpickle. Discards sequences that are shorter than seqmin 
           or contain characters other than the 20 amino acids.
           Stores whole protein info in self.proteins.
           
           Args:
               N/A
           Returns:
               N/A
        """
        allproteins = self.load_all_proteins()
        seqcount = 0
        while seqcount < self.numproteins:
            choice = np.random.randint(0,len(allproteins))
            protein = allproteins[choice]
            if (not protein.valid or len(protein.sequence) < self.seqmin):
                continue
            else:
                self.proteins.append(protein)
                seqcount += 1    
                
    def random_protein_search(self):
        """Finds epitopes for random proteins to generate statistics 
        
           Loops through all proteins and constructs a ProteinEpitopes object 
           and finds epitopes for each protein. It only saves the scores
           for each epitope, not the proteinepitopes or epitopes themselves.
           Converts scores to numpy arrays at end to make them 
           faster to use later. Pickle instance at end. Make sure kmerinfo
           has been initialized before pickling or you'll get an error.
           
           Args:
               N/A
           Returns:
               N/A
        """
        self.random_proteins()
        samplenums = []
        for sample in self.samples:
            samplenum = self.kmerinfo.sample2num[sample]
            samplenums.append(samplenum)
            self.samplescores[samplenum] = {}
            for epilen in range(self.epiminlen,self.epimaxlen+1):
                self.samplescores[samplenum][epilen] = [] 
        for protein in self.proteins:
            proteinepitopes = ProteinEpitopes(protein,self.kmerinfo,
                                             epiminlen=self.epiminlen,
                                             epimaxlen=self.epimaxlen,
                                             spanval=self.spanval,
                                             edgeweight=self.edgeweight)
            #Don't calculate statistics before they're created
            proteinepitopes.initialize_proteinepitopes(calcstatistics=False)      
            for samplenum in samplenums:
                for epitope in proteinepitopes.epitopedict[samplenum]:
                    epitopelen = len(epitope.sequence)
                    score = epitope.score
                    self.samplescores[samplenum][epitopelen].append(score)
        #Make scoresamples arrays for each length into np.array
        for samplenum in samplenums:
            for epilen in range(self.epiminlen,self.epimaxlen+1):
                scorearray = self.samplescores[samplenum][epilen]
                self.samplescores[samplenum][epilen] = np.array(scorearray)
        with open(self.picklename,'wb') as w:
            pickle.dump(self,w)    

    def scores_for_percentile(self,epitopethresh):
        """Returns the score corresponding to percentiles in scoresamples"""
        for samplenum in self.samplescores.keys():
            self.scorecutoffs[samplenum] = {}
            for epilen in range(self.epiminlen,self.epimaxlen+1):
                scorearray = self.samplescores[samplenum][epilen]
                score = np.percentile(scorearray,epitopethresh)
                score = float('{:.2f}'.format(score))
                self.scorecutoffs[samplenum][epilen] = score          
                                 

class ProteomicEpitopes:                                 
    """Generates epitope data for an entire proteome or any list of proteins
    
       Loops through each protein and stores epitope information.
       Stores epitopes with percentiles higher than epitopethresh.
       Contains other tools for analyzing the data.
       
       Attributes:
           kmerinfo ([KmerInfo]): kmer data used for calculations
           minprev (float): minimum fraction of samples epitope needs to be in
           epitopethresh (int): percentile cutoff for an epitope to be
              significant and in consensusepitopes or individualepitopes
           randomstats (RandomStats): contains statistical information.
               Initialized with object so it isn't loaded repetitively
           accessionfile (str): filename with uniprot accession on each line
           fastafile (str): filename with multiple proteins in fasta format              
           epiminlen (int): min length for epitopes
           epimaxlen (int): max length for epitopes
           specificsamples ([Sample]): list of samples to be analyzed. If not
               specificed, all samples are used   
           seqmin (int): minimum length of sequences used. Should be equal
               to at least 2 times the number of maximum positions. If sequence
               is too short, there is no way to tile it.                     
           randomseed (int): seed for random number generator          
           spanval (int): window size to average the frequency for each 
               position. Must be odd
           edgeweight (int): value that end positons in window are weighted.
               Positions closer to center are linearly weighted with center
               value being weighted as 1  
           accessions ([str]): list of accessions to be analyzed 
           proteins ([Protein]): proteins used in calculations
           consensusepitopes ([Epitope]): consensus epitopes compiled from
               multiple proteins
           individualepitopes ({samplenum}=[Epitope]): lists of top significant
               epitopes for each sample
           epitopescores = ({Epitope}{int}={str:float,str:float}):
               scores and percentiles indexed by epitope and samplenum.
               Keys are 'score' and 'percentile'               
           nextpercent (int): records progress in analyzing proteins
    """
    
    def __init__(self,kmerinfo,protdirectory,minprev,epitopethresh,
                 randomstats,accessionfile=None,fastafile=None,epiminlen=6,
                 epimaxlen=15,specificsamples=None,seqmin=10,spanval=7,
                 edgeweight=0.1):
        self.kmerinfo = kmerinfo   
        self.protdirectory = protdirectory
        self.minprev = minprev
        self.epitopethresh = epitopethresh
        self.randomstats = randomstats
        self.accessionfile = accessionfile
        self.fastafile = fastafile        
        self.epiminlen = epiminlen
        self.epimaxlen = epimaxlen  
        self.specificsamples = specificsamples
        if not self.specificsamples:
            self.specificsamples = self.kmerinfo.samples        
        self.seqmin = seqmin
        self.spanval = spanval
        self.edgeweight = edgeweight
        self.accessions = []
        self.proteins = []
        self.consensusepitopes = []
        self.individualepitopes = {}
        self.epitopescores = {}
        self.nextpercent = 0
        
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

    def __getstate__(self):
        """Makes sure large objects are not pickled"""
        #Make deep copy to not corrupt original
        state = dict(self.__dict__)
        del state['kmerinfo']
        del state['randomstats']
        del state['proteins']
        return state
    
    def load_proteins_accessions(self):
        """Gets accessions from file and loads all proteins"""
        if not self.accessionfile:
            raise ValueError('No accessionfile was provided')
        with open(self.accessionfile,'r') as f:   
            for line in f:        
                self.accessions.append(line.strip()) 
        for accession in self.accessions:
            #Will take far longer if none of these proteins in protdirectory
            protein = Protein(accession,self.protdirectory)
            protein.load_or_search()
            self.proteins.append(protein)
            
    def make_proteins_fasta(self):
        """Loads and pickles proteins from a fasta file
        
           If internet connection is an issue, it may be preferable to download
           a fastafile from uniprot.org. This function reads in proteins in a 
           similar manner as RandomStats. Pickles each protein when finished
           if not already pickled. After calling,  subsequent calls can be made
           with an accessionfile to load_proteins_accessions.
           
           Args:
               N/A
           Returns:
               N/A
        """
        if not self.fastafile:
            raise ValueError('No fastafile was provided')
        currentprot = ''
        with open(self.fastafile,'r') as f:
            for line in f:
                if line.startswith('>'):
                    if currentprot != '':
                        self.proteins.append(currentprot)
                        #Only pickle if it hasn't been pickled before
                        if not currentprot.load_protein():
                            currentprot.pickle_protein()
                    newprotein = Protein.parse_fasta_metadata(line,
                                              protdirectory=self.protdirectory)
                    newprotein.protdirectory = self.protdirectory
                    currentprot = newprotein
                else:
                    currentprot.add_to_seq(line)
            #Add last protein         
            self.proteins.append(currentprot)
        #Check whether proteins are valid
        for protein in self.proteins:
            protein.valid_characters()
            
    def proteomic_individual_epitopes(self):
        """Finds significant epitopes for each sample for a list of proteins
        
           Goes through each protein and calculated epitopes. Any epitopes
           that meet epitopethresh are kept. Each sample's epitopes are kept
           separately
           
           Args:
               N/A
           Returns:
               N/A
        """
        self.nextpercent = 10
        samplenums = []
        self.epitopescores = {}
        self.randomstats.scores_for_percentile(self.epitopethresh)
        scorecutoffs = self.randomstats.scorecutoffs        
        for sample in self.specificsamples:
            samplenum = self.kmerinfo.sample2num[sample]
            samplenums.append(samplenum)        
            self.individualepitopes[samplenum] = []
            self.epitopescores[samplenum] = {}
        for count,protein in enumerate(self.proteins):
            if (not protein.valid or len(protein.sequence) < self.seqmin):
                continue             
            proteinepitopes = ProteinEpitopes(protein,self.kmerinfo,
                                          epiminlen=self.epiminlen,
                                          epimaxlen=self.epimaxlen,
                                          randomstats=self.randomstats,
                                          specificsamples=self.specificsamples,
                                          spanval=self.spanval,
                                          edgeweight=self.edgeweight)
            proteinepitopes.initialize_proteinepitopes()  
            for samplenum in samplenums:
                epitopes = proteinepitopes.epitopedict[samplenum]
                proteinepitopes.add_epitopescores(epitopes)
                for epitope in epitopes:
                    epitopelen = len(epitope.sequence)
                    if epitope.score >= scorecutoffs[samplenum][epitopelen]:
                        self.individualepitopes[samplenum].append(epitope)
                        #Update epitopescores dict with individual scores
                        self.epitopescores[samplenum].update(proteinepitopes.
                                                             epitopescores)
            self.progress(count)
        #Sort by percentile
        for samplenum in samplenums:
            epitopes = self.individualepitopes[samplenum]
            self.individualepitopes[samplenum] = sorted(epitopes,key=lambda i: 
                                                     i.percentile,reverse=True)            
    
    def proteomic_consensus_epitopes(self):
        """Finds significant consensus epitopes for a list of proteins
        
           Goes through each protein and calculated epitopes. Then determines
           consensus epitopes that meet minprev and minthresh.
           
           Args:
               N/A
           Returns:
               N/A
        """        
        self.nextpercent = 10
        for count,protein in enumerate(self.proteins):
            if (not protein.valid or len(protein.sequence) < self.seqmin):
                continue             
            proteinepitopes = ProteinEpitopes(protein,self.kmerinfo,
                                          epiminlen=self.epiminlen,
                                          epimaxlen=self.epimaxlen,
                                          randomstats = self.randomstats,
                                          specificsamples=self.specificsamples)
            proteinepitopes.initialize_proteinepitopes()      
            proteinepitopes.consensus_epitopes(self.epitopethresh,
                                               self.minprev)
            self.consensusepitopes.extend(proteinepitopes.consensusepitopes)
            #Update epitopescores dict with individual protein epitopescores
            self.epitopescores.update(proteinepitopes.epitopescores)
            self.progress(count)
        #Sort by score than prevalence
        self.consensusepitopes = sorted(self.consensusepitopes,key=lambda i: 
                                        (i.prevalence,i.score),reverse=True)  
            
    def progress(self,count):
        """Prints percentage of proteins searched in multiples of 10% """
        totalprot = len(self.proteins)
        #Add 1 to count to compensate for indexing at 0
        percent = ((count+1)/totalprot)*100
        #If less than 10 proteins, just print fraction rounded to nearest 10
        if totalprot < 10:
            percent = int(round(percent,-1))
            print('Searched ' + str(percent) + '% of proteins',
                  flush=True)
        elif percent >= self.nextpercent:
            print('Searched ' + str(self.nextpercent) + '% of proteins',
                  flush=True)
            self.nextpercent += 10
     

class SpecificEpitope(Epitope):
    """Extends Epitope class to allow for specificity for comparing epitopes
    
       Attributes:
           protein (Protein): protein from which epitope was generated
           sequence (str): The amino acid sequence of epitope
           score (float): Numerical score of epitope from AUC of frequencies
           epirange (int,int): Is range that corresponds to epitope. It's
               (inclusive, exclusive)
           epitopevals (np.array): contains score for each position in epitope 
           percentile (float): percentage of random protein epitopes that have
               scores lower than this one's score
           centroid (float): center pos of epitope in entire protein sequence
           prevalence (float): percentage of population that reacts to epitope
           specificity (float): percentage of control population that doesn't
               react to the epitope
    """
    
    def __init__(self,epitope,specificity):
        super().__init__(epitope.protein,epitope.sequence,epitope.epirange,
                         epitope.epitopevals)
        self.prevalence = epitope.prevalence
        self.specificity = specificity
        
    def __repr__(self):
        """When object is printed, gives various initialized attributes"""
        s = self.sequence + ' ' + str(self.score) + ' ' + str(self.epirange)
        prevalence = str(float('{:.3f}'.format(self.prevalence)))
        specificity = str(float('{:.3f}'.format(self.specificity)))
        s += ' ' + 'Prevalence: ' + prevalence + ' '
        s += 'Specificity: ' + specificity + ' '
        s += self.protein.fullname + ' (' + self.protein.organism + ')'
        return s    


class GroupCompare:
    """Finds epitopes which are preferentially bound by experimental vs control
    
       Tiles protein sequence into kmers, then populates FrequencyInfo
       objects using individual or group scoring. Each sample has a separate
       FrequencyInfo object. Finds epitopes for each sample. Finds consensus
       epitopes for a population that meet a prevalence threshold. All 
       disease and control sample data should be in same enrichdicts.
       
       Attributes:
           protein (Protein): protein for which epitopes are being ofund
           kmerinfo (KmerInfo): contains enrichment info for kmers
           protdirectory (str): directory where protein pickles are found
           expthresh (int): percentile cutoff for an epitope to be
               significant in experimental group
           controlthresh (int): if an epitope exceeds this percentile in
               control specimens, epitope is not significant
           minprev (float): minimum fraction of samples epitope needs to
                   be in to be significant
           minspec (float): minimum specificity that an epitope must have
                   to be significant
           randomstats (RandomStats): randomstats file corresponding to data.
           expgroup ([Sample]): list of samples in experimental group
           controlgroup ([Sample]): list of samples in control group
           epiminlen (int): the minimum length of an epitope
           epimaxlen (int): the maximum length of an epitope      
           seqmin (int): minimum length of sequences used. Should be equal
               to at least 2 times the number of maximum positions. If sequence
               is too short, there is no way to tile it.                     
           spanval (int): window size to average the frequency for each 
               position. Must be odd
           edgeweight (int): value that end positons in window are weighted.
               Positions closer to center are linearly weighted with center
               value being weighted as 1   
           kmerinfocontrol (KmerInfo): enrichment info is control are in
               separate enrichdicts than experimental
           randomstatscontrol (RandomStats): randomstats file corresponding 
               control samplese if in separate enrichdicts than experimental
           proteins ([Protein]): all proteins analyzed
           specificepitopes ([SpecificEpitope]): epitopes found in Proteins
               that meet minprev, minspec, expthresh, and controlthresh
           epitopescores ({int}={str:float,str:float}): gives score and 
               percentile for each sample for the epitope. keys are 'score'
               and 'percentile'
    """    
    
    def __init__(self,kmerinfo,protdirectory,expthresh,
                 controlthresh,minprev,minspec,randomstats,expgroup,
                 controlgroup,epiminlen=6,epimaxlen=15,seqmin=10,spanval=7,
                 edgeweight=0.1,kmerinfocontrol=None,
                 randomstatscontrol=None):
        self.kmerinfo = kmerinfo   
        self.protdirectory = protdirectory
        self.expthresh = expthresh        
        self.controlthresh = controlthresh              
        self.minprev = minprev
        self.minspec = minspec
        self.randomstats = randomstats
        self.expgroup = expgroup
        self.controlgroup = controlgroup        
        self.epiminlen = epiminlen
        self.epimaxlen = epimaxlen  
        self.seqmin = seqmin
        self.spanval = spanval
        #If control samples come from different kmerInfos
        self.kmerinfocontrol = kmerinfocontrol
        self.randomstatscontrol = randomstatscontrol
        self.proteins = []
        self.edgeweight = edgeweight        
        self.specificepitopes = []
        self.epitopescores = {}
        
    def __getstate__(self):
        """Makes sure large objects are not pickled"""
        #Make deep copy to not corrupt original
        state = dict(self.__dict__)
        del state['kmerinfo']
        del state['kmerinfocontrol']
        del state['randomstatscontrol']
        del state['randomstats']
        del state['proteins']
        return state  
    
    def single_protein_compare(self,protein):
        """Identifies specificepitopes for a single protein
        
           Determines all epitopes for a single protein that meet  
           minprev, minspec, expthresh, and controlthresh. Stores these
           epitopes in specificepitopes attribute.
           
           Args:
               protein ([Protein]): Protein object to be analyzed
           Returns:
               N/A
        """
        self.proteins.append(protein)
        expinfo = ProteinEpitopes(protein,self.kmerinfo,
                                  randomstats=self.randomstats,
                                  specificsamples=self.expgroup)
        expinfo.initialize_proteinepitopes()
        if self.kmerinfocontrol:
            controlinfo = ProteinEpitopes(protein,self.kmerinfocontrol,
                                          randomstats=self.randomstatscontrol,
                                          specificsamples=self.controlgroup) 
        else:
            controlinfo = ProteinEpitopes(protein,self.kmerinfo,
                                          randomstats=self.randomstats,
                                          specificsamples=self.controlgroup) 
        controlinfo.initialize_proteinepitopes()        
        #Calculate consensus epitopes
        expinfo.consensus_epitopes(self.expthresh,self.minprev)
        if self.randomstatscontrol:
            self.randomstatscontrol.scores_for_percentile(self.controlthresh)
        else:
            self.randomstats.scores_for_percentile(self.controlthresh) 
        for epitope in expinfo.consensusepitopes:
            controlscores = controlinfo.group_scores(epitope)
            specificity = self.epitope_specificity(epitope,controlinfo,
                                                   controlscores)
            if specificity >= self.minspec:
                self.add_specificepitope(epitope,expinfo,controlscores,
                                         specificity)
        self.specificepitopes = sorted(self.specificepitopes,
                                       key=lambda i: i.score,reverse=True)
        
    def epitope_specificity(self,epitope,controlinfo,controlscores):
        """Determines the specificity for an epitope
        
           Uses randomstats scorecutoffs to determine the specificity of an
           epitope to determine if it meets the cutoff
           
           Args:
               epitope ([Epitope]): epitope to be evaluated
               controlinfo [ProteinEpitopes]: object containing epitope info
                   for control samples
               controlscores ({int}={str:float,str:float}): gives score and 
                 percentile for each sample for the epitope. keys are 'score'
                 and 'percentile'    
           Returns:
               specificity (float): the specificity of the input epitope
        """
        #Reference controlinfo's randomstats in case it uses different one
        scorecutoffs = controlinfo.randomstats.scorecutoffs
        epitopelen = len(epitope.sequence)
        numpresent = 0
        for samplenum in controlscores.keys():
            cutoff = scorecutoffs[samplenum][epitopelen]
            if controlscores[samplenum]['score'] >= cutoff:
                numpresent += 1
        specificity = 1 - (numpresent/len(controlinfo.samplenums))
        return specificity
    
    def add_specificepitope(self,epitope,expinfo,controlscores,specificity):
        """Adds an epitope that's been deemed specific to object
        
           Adds the SpecificEpitope to the specificepitopes attribute and 
           assigns scores for each sample for the epitope.
           
           Args:
               epitope ([Epitope]): the specific epitope to be added
               expinfo [ProteinEpitopes]: object containing epitope info
                   for experimental epitopes for determining score
               controlscores ({int}={str:float,str:float}): gives score and 
                 percentile for each sample for the epitope. keys are 'score'
                 and 'percentile'    
               specificity (float): the specificity of the input epitope
           Returns:
               N/A
        """
        epitopescores = {'exp':[],'control':[]}
        epitopescores['exp'] = expinfo.epitopescores[epitope]
        epitopescores['control'] = controlscores
        specificepitope = SpecificEpitope(epitope,specificity)
        self.specificepitopes.append(specificepitope)
        self.epitopescores[specificepitope] = epitopescores  
        
    def multiple_protein_compare(self,accessionfile=None,fastafile=None):
        """Determines specificepitopes using a collection of proteins
        
           Uses ProteomicEpitopes for experimental and control groups to
           identify specific epitopes for a collection of proteins. Can
           supply a FASTA file or a list of accessions, but not both.
           
           Args:
               accessionfile (str): filename with uniprot accession on each 
                   line
               fastafile (str): filename with multiple proteins in fasta format 
           Returns:
               N/A
        """
        #Make sure either an accessionfile or a fastafile are provided
        if accessionfile and fastafile:
            raise ValueError('Must supply accessionfile OR fastafile')
        if not accessionfile and not fastafile:
            raise ValueError('Must supply either an accessionfile or a '
                             'fastafile')
        if accessionfile:
            expinfo = ProteomicEpitopes(self.kmerinfo,self.protdirectory,
                                        self.minprev,self.expthresh,
                                        self.randomstats,
                                        accessionfile=accessionfile,
                                        specificsamples=self.expgroup)
            expinfo.load_proteins_accessions()
        if fastafile:
            expinfo = ProteomicEpitopes(self.kmerinfo,self.protdirectory,
                                        self.minprev,self.expthresh,
                                        self.randomstats,
                                        fastafile=fastafile,
                                        specificsamples=self.expgroup)  
            expinfo.make_proteins_fasta()
        self.proteins = expinfo.proteins
        expinfo.proteomic_consensus_epitopes()
        if self.randomstatscontrol:
            self.randomstatscontrol.scores_for_percentile(self.controlthresh)
        else:
            self.randomstats.scores_for_percentile(self.controlthresh)
        for epitope in expinfo.consensusepitopes:
            if self.kmerinfocontrol:
                controlinfo = ProteinEpitopes(epitope.protein,
                                          self.kmerinfocontrol,
                                          randomstats=self.randomstatscontrol,
                                          specificsamples=self.controlgroup)
            else:
                controlinfo = ProteinEpitopes(epitope.protein,self.kmerinfo,
                                             randomstats=self.randomstats,
                                             specificsamples=self.controlgroup) 
            controlinfo.initialize_proteinepitopes()     
            controlscores = controlinfo.group_scores(epitope)
            specificity = self.epitope_specificity(epitope,controlinfo,
                                                   controlscores)
            if specificity >= self.minspec:
                self.add_specificepitope(epitope,expinfo,controlscores,
                                         specificity)
        #Sort by sum of sensitivity and specificity
        self.specificepitopes = sorted(self.specificepitopes,
                                       key=lambda i: i.prevalence + 
                                       i.specificity,reverse=True)
        
        
class Similarity:
    """Allows for comparisons between similar sequences to quantify similarity
    
       Uses the PAM30 matrix to align and score two sequences or motifs with
       positions with brackets. Main use for kmer tiling is removing 
       redundancy in epitopes found in protein searches of 1,000s of proteins.
       
       Attributes:
           similarityfile (str): address of space-delimited similarity matrix,
               usually PAM30
           similarity ({str}{str}=int): Given 2 amino acids, gives the 
               similarity score
    """
    
    def __init__(self,similarityfile):
        self.similarityfile = similarityfile
        self.similarity = None
        self.load_similarity()

    def load_similarity(self):
        """Loads in similarity matrix from file"""
        similarity = {}
        with open(self.similarityfile,'r') as f:
            firstline = next(f)
            splitline = firstline.split()
            symbols = splitline[1:]
            for symbol in symbols:
                similarity[symbol] = {i:0 for i in symbols}
            for line in f:
                splitline = line.split()
                firstsymbol = splitline[0]
                for i,symbol in enumerate(symbols):
                    similarity[firstsymbol][symbol] = int(splitline[i+1])
        #Make X to X comparisons give score of 0
        similarity['X']['X'] = 0
        self.similarity = similarity 
        
    def proper_protein_name(self,epitope):
        """Removes extraneous strings from protein names"""
        if 'ECO' in epitope.protein.fullname:
            name = re.findall('(.+)(?={ECO)',
                              epitope.protein.fullname)[0].strip()
        else:       
             name = epitope.protein.fullname      
        return name          

    def convert_motif_list(self,motif):
        """Converts motif to list for easier processing"""
        listmotif = []
        bracketedaminos = re.findall('(?<=\[)([^\]]+)',motif)
        tabbedmotif = re.sub('[\[\]]', '\t', motif)
        for positions in tabbedmotif.split('\t'):
            if positions in bracketedaminos:
                listmotif.append(positions)
            else:
                for pos in list(positions):
                    listmotif.append(pos)
        return listmotif
                    
    def compare_motifs(self,motif,other):
        """Compares two sequences and generates a score
        
           Tries every alignment of two sequences and keeps the one with the
           highest score. Returns that high score. Gives same score if motif
           and other are reversed.
           
           Args:
               motif (str): first sequence for comparison
               other (str): second sequence for comparison
           Returns:
               bestscore (int): best similarity score of two motifs
        """
        listmotif = self.convert_motif_list(motif)
        motiflen = len(listmotif)
        listother = self.convert_motif_list(other)
        otherlen = len(listother)
        bestscore = -500
        beststart = -50
        #Add X's so that other can slide along motif
        motiffull = ['X']*(otherlen-1) + listmotif + ['X']*(otherlen-1)
        numalignments = motiflen + otherlen - 1
        #loop through all possible alignments
        for start in range(numalignments):
            otherfull = ['X']*start + listother 
            otherfull += ['X']*(len(motiffull) - start - otherlen)
            score = 0
            #Get score for this alignment
            for i,motifamino in enumerate(motiffull):
                otheramino = otherfull[i]
                if len(otheramino) == 1 and len(motifamino) == 1:
                      score += self.similarity[motifamino][otheramino]
                else:
                    brackscore = -500
                    brackcombos = itertools.product(motifamino,otheramino)
                    for combo in brackcombos:
                        comboscore = self.similarity[combo[0]][combo[1]]
                        if comboscore > brackscore:
                            brackscore = comboscore
                    score += brackscore
            if score > bestscore:
                bestscore = score
                beststart = start
        #Make it so that alignment where 1st position is coordinated is 0
        beststart -= otherlen - 1
        return bestscore    
    
    def non_redundant(self,epitopelist,similaritycutoff):
        """Given a list of epitopes, removes redundancy
        
           Uses similaritycutoff to determine if two epitopes are similar. 
           Useful when examining a number of similar proteins that may give
           similar epitopes. Goes linearly through the list and keeps earlier
           nonredundant epitopes. Specifically accounts for epitopes in
           the same protein, but in different species/strains. Also removes
           repetitive epitopes in the same protein.
           
           Args:
               epitopelist ([Epitope]): list of epitopes generated using 
                   KTOPE
               similaritycutoff (int): score above which epitopes are 
                   considered redundant with each other.
           Returns:
               nonredepitopes ([Epitope]): list of epitopes without any 
                   redundancy
        """        
        nonredepitopes = [epitopelist[0]]
        for otherepitope in epitopelist[1:]:
            otherseq = otherepitope.sequence
            othername = self.proper_protein_name(otherepitope)
            found = False
            for nonredepitope in nonredepitopes:
                nonredseq = nonredepitope.sequence
                nonredname = self.proper_protein_name(nonredepitope)
                #Group fragments together
                if (nonredname == (othername + ' (Fragment)') or
                    othername == (nonredname + ' (Fragment)')):
                        othername == nonredname                
                if nonredname == othername:
                    score = self.compare_motifs(nonredseq,otherseq)
                    if score > similaritycutoff:
                        found = True
            if not found:
                nonredepitopes.append(otherepitope)    
        return nonredepitopes    
 
    