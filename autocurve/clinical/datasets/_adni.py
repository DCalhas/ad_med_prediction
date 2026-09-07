from functools import reduce

import numpy as np

import pandas as pd

import os

from autocurve.clinical.datasets import _Dataset, _GetTransitions, lnds_indices, preprocess_step_curve, nearest_interp_dates

from autocurve.rna.datasets._adni import get_adni

__all__=["ADNI"]

class State():
	states=[0,1,2,3,4]
	values=[0,0.5,1,2,3]

	def __init__(self, x):
		assert x in State.values

		self.x=np.int32(State.values.index(x))

		if(self.x.shape): self.x=self.x[0]

	def __repr__(self,):

		return str(self.x)


class Curve(np.ndarray):
	"""
	Class for curve representation
	a curve has a t for timesteps and then it can have the measured values for p1, p1p, p2, p3 and hoehn&yahr score
	"""

	types=['CDMEMORY', 'CDORIENT', 'CDJUDGE', 'CDCOMMUN', 'CDHOME', 'CDCARE', ]
	"""
	CDMEMORY - Memory Score
	CDORIENT - Orientation Score
	CDJUDGE - Judgment and Problem Solving Score
	CDCOMMUN - Community Affairs Score
	CDHOME - Home and Hobbies Score
	CDCARE - Personal Care Score
	"""
	
	def __new__(cls, patno, t, birth, gender, cdr_features, cdr_values, cdr=None, ttol=4, **kwargs):
		
		cdr_features=cdr_features
		cdr_values = cdr_values.astype(np.int32)
		cdr = cdr.astype(np.float32)

		birth=birth.to_numpy()
		if(birth.shape and birth.shape[0]>0):
			birth=birth[0]
		else:
			birth=t.to_numpy()

		visit_age_weeks=(t.to_numpy()-birth).astype(np.timedelta64(1, 'W')).astype(np.float32)

		if(gender.shape and gender.shape[0]>0):
			gender=gender.to_numpy()[0].astype(np.int32)
		else:
			gender=-1

		_t=np.cumsum(np.concatenate((np.zeros((1,)), np.diff(t).astype(np.timedelta64(1, 'W')).astype(np.float32))))

		#obj=np.asarray(np.vstack([_t, visit_age_weeks, p1, p1p, p2, p3, hy])).view(cls)
		obj=np.asarray(np.vstack([cdr_values])).view(cls)

		cdr=np.nan_to_num(cdr, nan=0)
		cdr[np.where(cdr==101)]=0

		#preprocess curves
		cdr, _t, keep = preprocess_step_curve(
			cdr, _t, values=State.values, return_indices=True,  method="cummax",
		)
		visit_age_weeks = visit_age_weeks[keep]
		cond_measures = cond_measures[keep]
		dates = t.to_numpy().astype("datetime64[D]")[keep]

		obj.patno=patno
		obj.date=dates
		obj.t=_t
		obj.age_weeks=visit_age_weeks
		obj.gender=gender
		obj.cond_features=cdr_features #6 features!
		obj.cond_measures=cdr_values
		obj.progression=cdr
		types=dict(zip(Curve.types, len(Curve.types)*[True]))
		obj.types=types
		
		return obj

	def __array_finalize__(self, obj):
		if obj is None:
			return
		self.patno = getattr(obj, "patno", None)
		self.t = getattr(obj, "t", None)
		self.date=getattr(obj, "date", None)
		self.state_dbs=getattr(obj, "state_dbs", None)
		self.after_dbs=getattr(obj, "after_dbs", None)
		self.age_weeks=getattr(obj, "age_weeks", None)
		self.gender=getattr(obj, "gender", None)
		self.cond_measures = getattr(obj, "cond_measures", None)
		self.progression = getattr(obj, "progression", None)
		self.types = getattr(obj, "types", None)

	def getxy(self, curve_type="p1"):
		
		assert self.types[curve_type]

		return self.t, getattr(self, curve_type)


def _get_curves(dataset_dir="/ceph/hpc/data/s25r10-02-users/calhasd/adni", state="ON"):
	
	df=pd.read_csv(dataset_dir+"/clinical"+"/CDR_28Oct2025.csv")
	demog=pd.read_csv(dataset_dir+"/clinical/PTDEMOG_21Jan2026.csv", sep=",")
	
	individuals=np.unique(df['PTID'])

	pat_curves={}
	
	values=[101,0.0,0.5,1.0,2.0,3.0,np.nan]
	
	for i, ind in enumerate(individuals):
	
		ind_df=df[df['PTID']==ind].copy()
		ind_df=ind_df.sort_values('VISDATE').copy()
		ind_df['VISDATE']=ind_df['VISDATE'].str.strip()
		time=pd.to_datetime(ind_df['VISDATE'], format='%Y-%m-%d', errors='coerce')

		ind_demog=demog[demog['PTID']==ind]

		birth=pd.to_datetime(ind_demog['PTDOBYY'], format='%Y-%m-%d', errors='coerce')
		gender=ind_demog['PTGENDER']-1

		CDR=ind_df.copy()
		CDR=CDR.sort_values(by=['VISDATE',])
		CDR=CDR.drop(columns=CDR.columns.difference([col for col in CDR.columns if ("CD" in col) or col in ["PTID", "VISDATE", ]])).copy()
		cdrfeatures=[col for col in CDR.columns if "CD" in col  and not "CDRSB" in col and not "CDGLOBAL" in col and col not in ["ID", "VISDATE, PTID", "CDSOURCE", "CDVERSION",]]

		for col in cdrfeatures:
			CDR[col][CDR[col]==-1.0]=101
			CDR[col][CDR[col]>1.]=CDR[col][CDR[col]>1.]//1.#floor the value
			
			assert CDR[col].isin(values).all()
			CDR[col]=CDR[col].fillna(101)
		
		curve=Curve(ind, time, birth, gender, cdrfeatures, np.concatenate([np.expand_dims(CDR[col].to_numpy(), axis=-1) for col in cdrfeatures], axis=-1), ind_df['CDGLOBAL'].to_numpy(), ttol=1)
		
		if(curve is not None): pat_curves[ind]=curve
		print("I:", i, "of", individuals.shape[0], end="\r")
	
	print("I: Successfully read progression curves.", end="\n")
	return pat_curves


class Transition(np.ndarray):
	
	n_features=6
	n_values=7
	end_seq=6*np.ones((1,6)).astype(np.int32)
	max_len=50
	#[0,0.5,1,2,3,nan,endseq]
	#[0,1,2,3,4,5,6,,]
	
	def __new__(cls, patno : np.int32, duration : np.int32, start_age_weeks : np.int32, gender : np.int32, orig : State, dest : State, time : np.int32, measures : np.int32):

		obj=np.asarray([duration, start_age_weeks, gender, orig.x, dest.x, time]).view(cls).astype(np.int32)

		if(measures.shape[1]==1 and np.all(measures==101)): measures=np.repeat(measures,Transition.n_features,axis=1)

		obj.patno=patno
		obj.duration=duration
		obj.start_age_weeks=start_age_weeks
		obj.gender=gender
		obj.orig=orig.x
		obj.dest=dest.x
		obj.t=time
		obj.measures=measures
		obj.event=int(orig.x!=dest.x)
		obj.seq_len=obj.measures.shape[0]
		obj.measures=np.concatenate((obj.measures, Transition.end_seq), axis=0)#adjust seq len
		obj.measures[np.where(obj.measures==101)]=obj.n_values-2
		for s in range(len(State.states)):
			obj.measures[np.where(obj.measures==State.values[s])]=State.states[s]
		obj.measures=np.eye(obj.n_values)[obj.measures]
		
		#5 total features, 6 total values
		obj.measures=np.concatenate((obj.measures,np.zeros((Transition.max_len,obj.n_features,obj.n_values,))[:Transition.max_len-obj.measures.shape[0]]), axis=0) #pad sequence with zeros
		
		return obj

	def __array_finalize__(self, obj):
		if obj is None:
			return
		self.patno=getattr(obj, "patno", None)
		self.duration=getattr(obj, "duration", None)
		self.start_age_weeks=getattr(obj, "start_age_weeks", None)
		self.gender=getattr(obj, "gender", None)
		self.orig=getattr(obj, "orig", None)
		self.dest=getattr(obj, "dest", None)
		self.t=getattr(obj, "t", None)
		self.seq_len=getattr(obj, "seq_len", None)
		self.measures=getattr(obj, "measures", None)
		self.event=getattr(obj, "event", None)

adni_transitions=_GetTransitions(Transition, State, _get_curves)

class ADNI(_Dataset):

	get_dataset=get_adni
	curve=Curve
	state=State
	transition=Transition
	get_transitions=adni_transitions