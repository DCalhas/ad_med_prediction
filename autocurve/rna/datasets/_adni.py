import numpy as np

import pandas as pd

from autocurve.rna.datasets import _Dataset, _GetTransitions, preprocess_step_curve

__all__=["ADNI"]

def get_adni(dataset_dir="/ceph/hpc/data/s25r10-02-users/calhasd/adni"):
	
	adni_rna=pd.read_csv(dataset_dir+"/rna/ADNI_Gene_Expression_Profile.csv", low_memory=False)
	adni_rna_dict=pd.read_csv(dataset_dir+"/rna/ADNI_Gene_Expression_Profile_DICT.csv")
	
	individuals=np.array(adni_rna.iloc[1])[3:-1].astype("U25")
	
	GENEID_LIST=np.load(dataset_dir+"/rna/geneid.npy", allow_pickle=True)
	
	gene_ind_matrix=np.array(adni_rna[adni_rna['Unnamed: 2'].notna()].iloc[1:])[:,3:-1].astype(np.float32).T

	genes_names=np.array([g.split(" || ")[0] for g in np.array(adni_rna["Unnamed: 2"]).astype("U25") if g!='nan' and g!="Symbol"],).astype("U25")
	
	PAT_INFO=pd.read_csv(dataset_dir+"/clinical/"+"DXSUM_28Oct2025.csv", )
	total_target=np.array(PAT_INFO['DIAGNOSIS'])
	total_patients=np.array(PAT_INFO["PTID"]).astype("U25")
	
	targets=np.empty(individuals.shape, dtype=np.float32)
	targets[:]=np.nan
	for patidx in range(len(total_patients)):
		idx=np.where(individuals==total_patients[patidx])
		if(len(idx[0]) and ~np.isnan(total_target[patidx])):
			targets[idx]=int(total_target[patidx])-1
	targets=targets.astype(np.int32)
	
	genes_ids=GENEID_LIST[GENEID_LIST!=None]
	genes_names=genes_names[GENEID_LIST!=None]
	gene_ind_matrix=gene_ind_matrix[:,GENEID_LIST!=None]
	
	return gene_ind_matrix, genes_ids, genes_names, individuals, targets

class Curve(np.ndarray):
	"""
	Class for curve representation
	a curve has a t for timesteps and then it can have the measured values for p1, p1p, p2, p3 and hoehn&yahr score
	"""

	types=["cdr", "cdr_prog"]

	def __new__(cls, patno, t, birth, gender, cdr=None, cdr_prog=None, ttol=4, **kwargs):

		exists=[cdr is not None and len(cdr)>=ttol, cdr_prog is not None and len(cdr_prog)>=ttol]
		
		if(np.all(np.logical_not(exists))): return None
		
		cdr=cdr.astype(np.float32) if exists[0] else -1*np.ones((t.shape[0]))
		cdr_prog=cdr_prog.astype(np.float32) if exists[1] else -1*np.ones((t.shape[0]))

		birth=birth.to_numpy()
		if(birth.shape and birth.shape[0]>0):
			birth=birth[0]
		else:
			birth=t.to_numpy()#zero imputation
		visit_age_weeks=(t.to_numpy()-birth).astype(np.timedelta64(1, 'W')).astype(np.float32)

		if(gender.shape and gender.shape[0]>0):
			gender=gender.to_numpy()[0].astype(np.int32)
		else:
			gender=-1
		
		t=np.cumsum(np.concatenate((np.zeros((1,)), np.diff(t).astype(np.timedelta64(1, 'W')).astype(np.float32))))

		obj=np.asarray(np.vstack([t, visit_age_weeks, cdr_prog, cdr])).view(cls)

		cdr=np.nan_to_num(cdr)
		cdr_prog=np.nan_to_num(cdr_prog)

		obj.patno=patno

		#preprocess curves to be monotonically increasing
		cdr, t, keep = preprocess_step_curve(
			cdr, t, values=State.values, return_indices=True, method="cummax",
		)
		visit_age_weeks = visit_age_weeks[keep]
		cdr_prog = cdr_prog[keep]
		dates = t.to_numpy().astype("datetime64[D]")[keep]

		obj.t=t
		obj.age_weeks=visit_age_weeks
		obj.gender=gender
		obj.progression=cdr
		obj.cdr_prog=cdr_prog
		types=dict(zip(Curve.types, exists))
		obj.types=types

		return obj

	def __array_finalize__(self, obj):
		if obj is None:
			return
		self.patno=getattr(obj, "patno", None)
		self.t=getattr(obj, "t", None)
		self.age_weeks=getattr(obj, "age_weeks", None)
		self.gender=getattr(obj, "gender", None)
		self.cdr=getattr(obj, "cdr", None)
		self.cdr_prog=getattr(obj, "cdr_prog", None)
		self.types=getattr(obj, "types", None)

def _get_curves(dataset_dir="/ceph/hpc/data/s25r10-02-users/calhasd/adni", state=None):
	
	df=pd.read_csv(dataset_dir+"/clinical"+"/CDR_28Oct2025.csv")
	demog=pd.read_csv(dataset_dir+"/clinical/PTDEMOG_21Jan2026.csv", sep=",")
	
	individuals=np.unique(df['PTID'])

	pat_curves={}
	
	for i, ind in enumerate(individuals):
	
		ind_df=df[df['PTID']==ind].copy()
		ind_df=ind_df.sort_values('VISDATE').copy()
		ind_df['VISDATE']=ind_df['VISDATE'].str.strip()
		time=pd.to_datetime(ind_df['VISDATE'], format='%Y-%m-%d', errors='coerce')

		ind_demog=demog[demog['PTID']==ind]

		birth=pd.to_datetime(ind_demog['PTDOBYY'], format='%Y-%m-%d', errors='coerce')
		gender=ind_demog['PTGENDER']-1
		
		curve=Curve(ind, time, birth, gender, ind_df['CDGLOBAL'].to_numpy(), ind_df['CDRSB'].to_numpy(), ttol=1)
		if(curve is not None): pat_curves[ind]=curve
		print("I:", i, "of", individuals.shape[0], end="\r")

	print("I: Successfully read progression curves", end="\r")

	return pat_curves


class State():
	
	values=[0.,0.5,1.,2.,3.,]
	states=[0,1,2,3,4,]

	def __init__(self, x):
		
		assert x in State.values

		self.x=np.int32(State.states[State.values.index(x)])

		if(self.x.shape): self.x=self.x[0]

	def __repr__(self,):

		return str(self.x)

class Transition(np.ndarray):

	def __new__(cls, patno : np.int32, duration : np.int32, gender : np.int32, orig : State, dest : State, time : np.int32):
		obj=np.asarray([duration, gender, orig.x, dest.x, time]).view(cls).astype(np.int32)

		obj.patno=patno
		obj.duration=duration		
		obj.gender=gender
		obj.orig=orig.x
		obj.dest=dest.x
		obj.t=time
		obj.event=int(orig.x!=dest.x)
		
		return obj

	def __array_finalize__(self, obj):
		if obj is None:
			return
		self.patno=getattr(obj, "patno", None)
		self.duration=getattr(obj, "duration", None)		
		self.gender=getattr(obj, "gender", None)
		self.orig=getattr(obj, "orig", None)
		self.dest=getattr(obj, "dest", None)
		self.t=getattr(obj, "t", None)
		self.event=getattr(obj, "event", None)


adni_transitions=_GetTransitions(Transition, State, _get_curves)

class ADNI(_Dataset):

	get_dataset=get_adni
	curve=Curve
	state=State
	transition=Transition
	get_transitions=adni_transitions