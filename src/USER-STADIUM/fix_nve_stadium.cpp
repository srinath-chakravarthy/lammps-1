/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fix_nve_stadium.h"
#include "atom.h"
#include "force.h"
#include "update.h"
#include "respa.h"
#include "modify.h"
#include "comm.h"
#include "error.h"
#include "domain.h"
#include "modify.h"
#include "region.h"
#include "input.h"
#include "variable.h"
#include "random_mars.h"
#include "memory.h"
#include "math_extra.h"
#include "group.h"
#include "comm.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace FixConst;

enum{NOBIAS,BIAS};
enum{CONSTANT,EQUAL,ATOM};


/* ---------------------------------------------------------------------- */

FixNVEStadium::FixNVEStadium(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg != 15) error->all(FLERR,"Illegal fix nve/langevin command");

  time_integrate = 1;
  scalar_flag = 1;
  global_freq = 1;
  extscalar = 1;

  ncount = 0;

  nevery = 1;

  
  /* Flags to output per atom quantities 
   */
  peratom_flag = 1; /// Sets per atom quantity output
  peratom_freq = 1; /// Set per atom output frequency
  
  /* gamma_stadium now stores the previous acceleration */
  size_peratom_cols = 4; /// Sets the size of per atom arrays/vectors
  
  
  
  /* Flags for restart
   */ 
  restart_peratom = 1; /// Save per atom state to restart file
  
  tstr = NULL;
  if (strstr(arg[3],"v_") == arg[3]) {
    int n = strlen(&arg[3][2]) + 1;
    tstr = new char[n];
    strcpy(tstr,&arg[3][2]);
  } else {
    t_start = force->numeric(FLERR,arg[3]);
    t_target = t_start;
    tstyle = CONSTANT;
  }

  t_stop = force->numeric(FLERR,arg[4]);
  t_period = force->numeric(FLERR,arg[5]);
  seed = force->inumeric(FLERR,arg[6]);

  if (t_period <= 0.0) error->all(FLERR,"Fix langevin period must be > 0.0");
  if (seed <= 0) error->all(FLERR,"Illegal fix langevin command");

  // initialize Marsaglia RNG with processor-unique seed

  random = new RanMars(lmp,seed + comm->me);

  // allocate per-type arrays for force prefactors

  gfactor1 = new double[atom->ntypes+1];
  gfactor2 = new double[atom->ntypes+1];
  ratio = new double[atom->ntypes+1];

  for (int i = 1; i <= atom->ntypes; i++) ratio[i] = 1.0;
  
  int iarg = 7;
  while(iarg < narg) {
    /*  Stadium Langevin part of thermostat 
	Check if flag is set to yes or no
	if no, then do not check for anything else
	if yes -> then check to make sure other arguments exist and need to exist
	these are xmin, xmax, ymin, ymax and width
    */
    if (iarg+8 > narg) error->all(FLERR,"Illegal fix langevin command (stadium)");
    s_xmin = force->numeric(FLERR,arg[iarg+1]);
    s_xmax = force->numeric(FLERR,arg[iarg+2]);
    s_ymin = force->numeric(FLERR,arg[iarg+3]);
    s_ymax = force->numeric(FLERR,arg[iarg+4]);
    s_zmin = force->numeric(FLERR,arg[iarg+5]);
    s_zmax = force->numeric(FLERR,arg[iarg+6]);
    s_width = force->numeric(FLERR,arg[iarg+7]);
    if (s_xmin > s_xmax) error->all(FLERR, "Illegal stadium langevin, xmax must be greater than xmin");
    if (s_ymin > s_ymax) error->all(FLERR, "Illegal stadium langevin, ymax must be greater than ymin");
    if (s_zmin > s_zmax) error->all(FLERR, "Illegal stadium langevin, zmax must be greater than zmin");
    if (s_width < 0.0) error->all(FLERR, "Illegal stadium langevin, width must be greater than 0.0");
    stadflag = 1;
    iarg += 8;
  }

  // set temperature = NULL, user can override via fix_modify if wants bias

  id_temp = NULL;
  temperature = NULL;

  maxatom1 = maxatom2 = 0;
  
  flangevin = NULL;
  nvalues = 4;
  grow_arrays(atom->nmax);
  atom->add_callback(0);  /// Call back for new to grow arrays
  atom->add_callback(1);  /// Call back for restart 
    
  // Initialize the stadium langevin peratom damping coefficient factor
  // 	values are issued as scale factors which are equal to 1.0 - abs(min(x-xmin, x-xmax, y-ymin, y-max)
  int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++){
    flangevin[i][0] = 0.0;
    flangevin[i][1] = 0.0;
    flangevin[i][2] = 0.0;
    flangevin[i][3] = 0.0; 
  }
  // Stadium langevin initiliazation
  double **x = atom->x;
  int *mask = atom->mask;
  for (int i = 0; i < nlocal; i++){
    if (mask[i] & groupbit) {
      double x1 = fabs(x[i][0] - s_xmin);
      double x2 = fabs(x[i][0] - s_xmax);
      double y1 = fabs(x[i][1] - s_ymin);
      double y2 = fabs(x[i][1] - s_ymax);
      double z1 = fabs(x[i][2] - s_zmin);
      double z2 = fabs(x[i][2] - s_zmax);

      double dxy = (minvalue(x1, x2, y1, y2, z1, z2));
      if (dxy > s_width)
	{
	  flangevin[i][0] = 0.0;
	} else
	{
	  flangevin[i][0] = 1.0 - dxy/s_width;
	}
    } else
      {
	flangevin[i][0] = 0.0;
      }
  }
  nrestart = 1;
  nrestart = 5;

}
/* ---------------------------------------------------------------------- */
FixNVEStadium::~FixNVEStadium()
{
  delete random;
  delete [] tstr;
  delete [] gfactor1;
  delete [] gfactor2;
  delete [] ratio;
  delete [] id_temp;
  memory->destroy(flangevin);
  atom->delete_callback(id,0);
  atom->delete_callback(id,1);

}


/* ---------------------------------------------------------------------- */

int FixNVEStadium::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  mask |= INITIAL_INTEGRATE_RESPA;
  mask |= FINAL_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixNVEStadium::init()
{
  double fdrag[3], fran[3];
  double boltz = force->boltz;
  double dt = update->dt;
  double mvv2e = force->mvv2e;
  double ftm2v = force->ftm2v;

  //compute_target();

  
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;

  if (strstr(update->integrate_style,"respa"))
    step_respa = ((Respa *) update->integrate)->step;

  // warn if using fix shake, which will lead to invalid constraint forces

  for (int i = 0; i < modify->nfix; i++)
    if ((strcmp(modify->fix[i]->style,"shake") == 0)
        || (strcmp(modify->fix[i]->style,"rattle") == 0)) {
      if (comm->me == 0)
        error->warning(FLERR,"Should not use fix nve/limit with fix shake or fix rattle");
    }
  if (!atom->rmass) {
    for (int i = 1; i <= atom->ntypes; i++) {
      gfactor1[i] = -atom->mass[i] / t_period / force->ftm2v;
      gfactor2[i] = sqrt(atom->mass[i]) *
        sqrt(24.0*force->boltz/t_period/update->dt/force->mvv2e) /
        force->ftm2v;
      gfactor1[i] *= 1.0/ratio[i];
      gfactor2[i] *= 1.0/sqrt(ratio[i]);
    }
  }

}

/* ----------------------------------------------------------------------
   allow for both per-type and per-atom mass
------------------------------------------------------------------------- */

void FixNVEStadium::initial_integrate(int vflag)
{
  double dtfm,vsq,scale;

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  // Langevin terms
  double fdrag[3], fran[3], v_orig[3];
  double gamma1, gamma2;

  double boltz = force->boltz;
  double dt = update->dt;
  double mvv2e = force->mvv2e;
  double ftm2v = force->ftm2v;

  int atomstyle=(tstyle==ATOM);

  /// Compute target temperature ....
  compute_target();

  
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] && groupbit) {
      if (atomstyle) tsqrt = sqrt(tforce[i]);
      if (rmass) {
	gamma1 = -rmass[i] / t_period / ftm2v;
	gamma2 = sqrt(rmass[i]) * sqrt(24.0*boltz/t_period/dt/mvv2e) / ftm2v;
	gamma1 *= 1.0/ratio[type[i]];
	dtfm = dtf/rmass[i];
      }
      else {
	gamma1 = gfactor1[type[i]];
	gamma2 = gfactor2[type[i]] * tsqrt;
	dtfm = dtf / mass[type[i]];
      }
	
      gamma1 *= flangevin[i][0];
      gamma2 *= sqrt(flangevin[i][0]);

      if (flangevin[i][0] > 1.e-6) {
	v[i][0] += dtfm * (f[i][0] + flangevin[i][1]);
	v[i][1] += dtfm * (f[i][1] + flangevin[i][2]);
	if (s_zmax > 100000.0 && s_zmin < -100000.00) {
	  v[i][2] = dtfm*(f[i][2]);
	} else { 
	  v[i][2] += dtfm * (f[i][2] + flangevin[i][3]);
	}
      }
      else {
	v[i][0] += dtfm * f[i][0];
	v[i][1] += dtfm * f[i][1];
	v[i][2] += dtfm * f[i][2];
      }
      // Update the positions after initial integration
      x[i][0] += dtv * v[i][0];
      x[i][1] += dtv * v[i][1];
      x[i][2] += dtv * v[i][2];
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixNVEStadium::final_integrate()
{
  double dtfm,vsq,scale;

  double **v = atom->v;
  double **f = atom->f;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  // Langevin terms
  double fdrag[3], fran[3];
  double gamma1, gamma2;

  double boltz = force->boltz;
  double dt = update->dt;
  double mvv2e = force->mvv2e;
  double ftm2v = force->ftm2v;

  int atomstyle=(tstyle==ATOM);

  /// Compute target temperature ....
  //compute_target();
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] && groupbit) {
      if (atomstyle) tsqrt = sqrt(tforce[i]);
      if (rmass) {
	gamma1 = -rmass[i] / t_period / ftm2v;
	gamma2 = sqrt(rmass[i]) * sqrt(24.0*boltz/t_period/dt/mvv2e) / ftm2v;
	gamma1 *= 1.0/ratio[type[i]];
	dtfm = dtf/rmass[i];
      }
      else {
	gamma1 = gfactor1[type[i]];
	gamma2 = gfactor2[type[i]] * tsqrt;
	dtfm = dtf / mass[type[i]];
      }
      gamma1 *= flangevin[i][0];
      gamma2 *= sqrt(flangevin[i][0]);

      fran[0] = gamma2*(random->uniform()-0.5);
      fran[1] = gamma2*(random->uniform()-0.5);
      fran[2] = gamma2*(random->uniform()-0.5);

      fdrag[0] = gamma1*v[i][0];
      fdrag[1] = gamma1*v[i][1];
      fdrag[2] = gamma1*v[i][2];
      
      if (flangevin[i][0] > 1.e-6) {
	v[i][0] += dtfm * (f[i][0] + fran[0] + fdrag[0]);
	v[i][1] += dtfm * (f[i][1] + fran[1] + fdrag[1]);

	flangevin[i][1] = fran[0] + fdrag[0];
	flangevin[i][2] = fran[1] + fdrag[1];
	if (s_zmax > 100000.0 && s_zmin < 100000.0) {
		v[i][2] += dtfm * f[i][2];
		flangevin[i][3] = 0.0;
	} else {
		v[i][2] += dtfm * (f[i][2] + fran[2] + fdrag[2]);
		flangevin[i][3] = fran[2] + fdrag[2];	  
	}
      }
      else {
	v[i][0] += dtfm * f[i][0];
	v[i][1] += dtfm * f[i][1];
	v[i][2] += dtfm * f[i][2];	  
      }
      
    }
  }  
}

/* ---------------------------------------------------------------------- */

void FixNVEStadium::initial_integrate_respa(int vflag, int ilevel, int iloop)
{
  dtv = step_respa[ilevel];
  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;

  if (ilevel == 0) initial_integrate(vflag);
  else final_integrate();
}

/* ---------------------------------------------------------------------- */

void FixNVEStadium::final_integrate_respa(int ilevel, int iloop)
{
  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;
  final_integrate();
}

/* ---------------------------------------------------------------------- */

void FixNVEStadium::reset_dt()
{
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
  
  if (atom->mass) {
    for (int i = 1; i <= atom->ntypes; i++) {
      gfactor2[i] = sqrt(atom->mass[i]) *
        sqrt(24.0*force->boltz/t_period/update->dt/force->mvv2e) /
        force->ftm2v;
      gfactor2[i] *= 1.0/sqrt(ratio[i]);
    }
  }
}

/* ----------------------------------------------------------------------
   energy of indenter interaction
------------------------------------------------------------------------- */

double FixNVEStadium::compute_scalar()
{
  double one = ncount;
  double all;
  MPI_Allreduce(&one,&all,1,MPI_DOUBLE,MPI_SUM,world);
  return all;
}

/* ----------------------------------------------------------------------
   set current t_target and t_sqrt
------------------------------------------------------------------------- */

void FixNVEStadium::compute_target()
{
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double delta = update->ntimestep - update->beginstep;
  if (delta != 0.0) delta /= update->endstep - update->beginstep;

  // if variable temp, evaluate variable, wrap with clear/add
  // reallocate tforce array if necessary

  if (tstyle == CONSTANT) {
    t_target = t_start + delta * (t_stop-t_start);
    tsqrt = sqrt(t_target);
  } else {
    modify->clearstep_compute();
    if (tstyle == EQUAL) {
      t_target = input->variable->compute_equal(tvar);
      if (t_target < 0.0)
        error->one(FLERR,"Fix langevin variable returned negative temperature");
      tsqrt = sqrt(t_target);
    } else {
      if (nlocal > maxatom2) {
        maxatom2 = atom->nmax;
        memory->destroy(tforce);
        memory->create(tforce,maxatom2,"langevin:tforce");
      }
      input->variable->compute_atom(tvar,igroup,tforce,1,0);
      for (int i = 0; i < nlocal; i++)
        if (mask[i] & groupbit)
            if (tforce[i] < 0.0)
              error->one(FLERR,
                         "Fix langevin variable returned negative temperature");
    }
    modify->addstep_compute(update->ntimestep + 1);
  }
}

/* ----------------------------------------------------------------------
   extract thermostat properties
------------------------------------------------------------------------- */

void *FixNVEStadium::extract(const char *str, int &dim)
{
  dim = 0;
  if (strcmp(str,"t_target") == 0) {
    return &t_target;
  }
  return NULL;
}

/* ----------------------------------------------------------------------
   memory usage of langevin array
------------------------------------------------------------------------- */

double FixNVEStadium::memory_usage()
{
  double bytes = 0.0;
  // Stadium Langevin memory usage
  bytes += atom->nmax *4* sizeof(double);
  return bytes;
}


/* ----------------------------------------------------------------------
   allocate atom-based array for stadium
------------------------------------------------------------------------- */

void FixNVEStadium::grow_arrays(int nmax)
{
  memory->grow(flangevin,nmax,4,"fix_nve_stadidum:flangevin");
  array_atom = flangevin;
}
/* ----------------------------------------------------------------------
   copy values within local atom-based array
------------------------------------------------------------------------- */

void FixNVEStadium::copy_arrays(int i, int j, int delflag)
{
  for (int m=0; m<nvalues; m++)
    flangevin[j][m] = flangevin[i][m];
}

/* ----------------------------------------------------------------------
   pack values in local atom-based array for exchange with another proc
------------------------------------------------------------------------- */

int FixNVEStadium::pack_exchange(int i, double *buf)
{
  int n = 0;
  for (int m =0; m < nvalues; m++) {
    buf[n++] = flangevin[i][m];
  }
  return n;

}

/* ----------------------------------------------------------------------
   unpack values in local atom-based array from exchange with another proc
------------------------------------------------------------------------- */

int FixNVEStadium::unpack_exchange(int nlocal, double *buf)
{
  int n = 0;
  for (int m =0; m < nvalues; m++){
    flangevin[nlocal][m] = buf[n++];
  }
  return n;   
}

/* ----------------------------------------------------------------------
   pack values in local atom-based arrays for restart file
------------------------------------------------------------------------- */

int FixNVEStadium::pack_restart(int i, double *buf)
{
  buf[0] = 5;
  buf[1] = flangevin[i][0];
  buf[2] = flangevin[i][1];
  buf[3] = flangevin[i][2];
  buf[4] = flangevin[i][3];
  return 5;
}

/* ----------------------------------------------------------------------
   unpack values from atom->extra array to restart the fix
------------------------------------------------------------------------- */

void FixNVEStadium::unpack_restart(int nlocal, int nth)
{
  double **extra = atom->extra;

  // skip to Nth set of extra values

  int m = 0;
  for (int i = 0; i < nth; i++) m += static_cast<int> (extra[nlocal][m]);
  m++;

  flangevin[nlocal][0] = extra[nlocal][m++];
  flangevin[nlocal][1] = extra[nlocal][m++];
  flangevin[nlocal][2] = extra[nlocal][m++];
  flangevin[nlocal][3] = extra[nlocal][m++];
}


/* ----------------------------------------------------------------------
   maxsize of any atom's restart data
------------------------------------------------------------------------- */

int FixNVEStadium::maxsize_restart()
{
  return 5;
}

/* ----------------------------------------------------------------------
   size of atom nlocal's restart data
------------------------------------------------------------------------- */

int FixNVEStadium::size_restart(int nlocal)
{
  return 5;
}


/* ----------------------------------------------------------------------
   Find minumum values of 4 given values 
------------------------------------------------------------------------- */
double FixNVEStadium::minvalue(double A, double B, double C, double D)
{
  double minval = 1.0e20;
  if (A < minval){
    minval = A;
  }
  if (B < minval){
    minval = B;
  }
  if (C < minval){
    minval = C;
  }
  if (D < minval){
    minval = D;
  }
  return minval; 
}

/* ----------------------------------------------------------------------
 * Overloaded 
   Find minumum values of 6 given values 
------------------------------------------------------------------------- */
double FixNVEStadium::minvalue(double A, double B, double C, double D, double E, double F)
{
  double minval = 1.0e20;
  if (A < minval){
    minval = A;
  }
  if (B < minval){
    minval = B;
  }
  if (C < minval){
    minval = C;
  }
  if (D < minval){
    minval = D;
  }
  if (E < minval){
    minval = E;
  }
  if (F < minval){
    minval = F;
  }
  return minval; 
}
