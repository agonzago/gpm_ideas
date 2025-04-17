module uncmin_int
contains
subroutine uncmin ( n, x0, fcn, x, f, info, w, lw )

!*****************************************************************************80
!
!! UNCMIN minimizes a smooth nonlinear function of N variables.
!
!  Discussion:
!
!    A subroutine that computes the function value at any point
!    must be supplied.  Derivative values are not required.
!    this subroutine provides the simplest interface to the uncmin
!    minimization package.  The user has no control over options.
!
!    This routine uses a quasi-Newton algorithm with line search
!    to minimize the function represented by the subroutine fcn.
!    at each iteration, the nonlinear function is approximated
!    by a quadratic function derived from a taylor series.
!    the quadratic function is minimized to obtain a search direction,
!    and an approximate minimum of the nonlinear function along
!    the search direction is found using a line search.  the
!    algorithm computes an approximation to the second derivative
!    matrix of the nonlinear function using quasi-Newton techniques.
!
!    The uncmin package is quite general, and provides many options
!    for the user.  however, this subroutine is designed to be
!    easy to use, with few choices allowed.  for example:
!
!    1.  only function values need be computed.  first derivative
!    values are obtained by finite differencing.  this can be
!    very costly when the number of variables is large.
!
!    2.  it is assumed that the function values can be obtained
!    accurately (to an accuracy comparable to the precision of
!    the computer arithmetic).
!
!    3.  at most 150 iterations are allowed.
!
!    4.  it is assumed that the function values are well-scaled,
!    that is, that the optimal function value is not pathologically
!    large or small.
!
!  Reference:
!
!    John Dennis, Robert Schnabel,
!    Numerical Methods for Unconstrained Optimization
!    and Nonlinear Equations,
!    SIAM, 1996,
!    ISBN13: 978-0-898713-64-0,
!    LC: QA402.5.D44.
!
!    Robert Schnabel, John Koontz, and B E Weiss,
!    A modular system of algorithms for unconstrained minimization,
!    Report cu-cs-240-82,
!    Computer Science Department,
!    University of Colorado at Boulder, 1982.
!
!  Parameters:
!
!    Input, integer N, the dimension of the problem.
!
!    Input, real ( kind = 8 ) X0(N), an initial estimate of the minimizer.
!
!    Input, external FCN, the name of the routine to evaluate the minimization
!    function, of the form
!
!      subroutine fcn ( n, x, f )
!      integer n
!      real ( kind = 8 ) x(n)
!      real ( kind = 8 ) f
!
!    Output, real ( kind = 8 ) X(N), the local minimizer.
!
!    Output, real ( kind = 8 ) F, the function value at X.
!
!    Output, integer INFO, termination code.
!    0:  optimal solution found.
!    1:  terminated with gradient small, X is probably optimal.
!    2:  terminated with stepsize small, X is probably optimal.
!    3:  lower point cannot be found, X is probably optimal.
!    4:  iteration limit (150) exceeded.
!    5:  too many large steps, function may be unbounded.
!    -1:  insufficient workspace.
!
!    Workspace, real ( kind = 8 ) W(LW).
!
!    Input, integer LW, the size of the workspace vector W, which
!    must be at least N * ( N + 10 ).
!
  implicit none

  integer lw
  integer n

  external d1fcn
  external d2fcn
  real ( kind = 8 ) dlt
  real ( kind = 8 ) epsm
  real ( kind = 8 ) f
  external fcn
  real ( kind = 8 ) fscale
  real ( kind = 8 ) gradtl
  integer ia
  integer iagflg
  integer iahflg
  integer iexp
  integer ig
  integer info
  integer ipr
  integer it
  integer itnlim
  integer iw1
  integer iw2
  integer iw3
  integer iw4
  integer iw5
  integer iw6
  integer iw7
  integer iw8
  integer lwmin
  integer method
  integer msg
  integer ndigit
  integer nr
  real ( kind = 8 ) stepmx
  real ( kind = 8 ) steptl
  real ( kind = 8 ) w(lw)
  real ( kind = 8 ) x(n)
  real ( kind = 8 ) x0(n)
!
!  Subdivide workspace
!
  ig  = 1
  it  = ig  + n
  iw1 = it  + n
  iw2 = iw1 + n
  iw3 = iw2 + n
  iw4 = iw3 + n
  iw5 = iw4 + n
  iw6 = iw5 + n
  iw7 = iw6 + n
  iw8 = iw7 + n
  ia  = iw8 + n
  lwmin = ia + n*n-1

  if ( lw < lwmin ) then
    info = -1
    write ( *, '(a)' ) ' '
    write ( *, '(a)' ) 'UNCMIN - Fatal error!'
    write ( *, '(a)' ) '  Insufficient workspace.'
    write ( *, '(a)' ) '  LW < LWMIN.'
    write ( *, '(a,i6)' ) '  LW = ', lw
    write ( *, '(a,i6)' ) '  LWMIN = ', lwmin
    stop
  end if
!
!  Set up parameters for OPTDRV.
!
!  parameters that should not be changed when using condensed code
!
! nr     = parameter used to divide workspace
! method = 1 (line search) -- do not change
! msg    = 9 => no printing, n=1 allowed
! iahflg = 1 => analytic hessian  supplied (0 otherwise)
! ipr    = device for output (irrelevant in current version)
! dlt    = (irrelevant parameter for method = 1)
! epsm   = machine epsilon
!
  nr = n
  method = 1
  msg = 9
  iahflg = 0
  ipr = 6
  dlt = -1.0D+00
  epsm = epsilon ( epsm )
!
! parameters that may be changed:
!
! iexp   = 1 means function expensive to evaluate (iexp = 0 => cheap)
! iagflg = 1 means analytic gradient supplied (0 otherwise)
! ndigit = -1 means optdrv assumes f is fully accurate
! itnlim = 150 = maximum number of iterations allowed
! gradtl = zero tolerance for gradient, for convergence tests
! stepmx = maximum allowable step size
! steptl = zero tolerance for step, for convergence tests
! fscale = typical order-of-magnitude size of function
! typsiz = typical order-of-magnitude size of x (stored in w(lt))
!
  iexp = 1
  iagflg = 0
  ndigit = -1
  itnlim = 1000
  
  gradtl = epsm**(1.0D+00 / 3.0D+00 )
  stepmx = 0.0D+00
  steptl = sqrt ( epsm )
  fscale = 1.0D+00
  w(it:it+n-1) = 1.0D+00
!
!  Minimize function
!
  call optdrv ( nr, n, x0, fcn, d1fcn, d2fcn, w(it), fscale, method, iexp, &
    msg, ndigit, itnlim, iagflg, iahflg,  ipr, dlt, gradtl, stepmx, steptl, &
    x, f, w(ig), info, w(ia), w(iw1), w(iw2), w(iw3), w(iw4), &
    w(iw5), w(iw6), w(iw7), w(iw8))

  if ( info == 1 ) then
    write ( *, '(a)' ) ' '
    write ( *, '(a)' ) 'UNCMIN - Note!'
    write ( *, '(a)' ) '  INFO = 1.'
    write ( *, '(a)' ) '  The iteration probably converged.'
    write ( *, '(a)' ) '  The gradient is very small.'
    return
  end if

  if ( info == 2 ) then
    write ( *, '(a)' ) ' '
    write ( *, '(a)' ) 'UNCMIN - Note!'
    write ( *, '(a)' ) '  INFO = 2.'
    write ( *, '(a)' ) '  The iteration probably converged.'
    write ( *, '(a)' ) '  The stepsize is very small.'
    return
  end if

  if ( info == 3 ) then
    write ( *, '(a)' ) ' '
    write ( *, '(a)' ) 'UNCMIN - Warning!'
    write ( *, '(a)' ) '  INFO = 3.'
    write ( *, '(a)' ) '  Cannot find a point with lower value.'
    write ( *, '(a)' ) '  (But not completely happy with the current value.)'
    return
  end if

  if ( info == 4 ) then
    write ( *, '(a)' ) ' '
    write ( *, '(a)' ) 'UNCMIN - Warning!'
    write ( *, '(a)' ) '  INFO = 4.'
    write ( *, '(a)') '  Too many iterations.'
    return
  end if

  if ( info == 5 ) then
    write ( *, '(a)' ) ' '
    write ( *, '(a)' ) 'UNCMIN - Warning!'
    write ( *, '(a)' ) '  INFO = 5.'
    write ( *, '(a)' ) '  Too many large steps.'
    write ( *, '(a)' ) '  The function may be unbounded.'
    return
  end if

  return
end subroutine uncmin
subroutine bakslv ( nr, n, a, x, b )

!*****************************************************************************80
!
!! BAKSLV solves A'*x=b where A is a lower triangular matrix.
!
!  Discussion:
!
!    BAKSLV solves the linear equations A'*X=B, where A is a
!    lower triangular matrix and A' is the transpose of A.
!
!    This routine is required by the UNCMIN minimization program.
!
!    If B is no longer required by calling routine, then vectors B and
!    X may share the same storage, and the output value of X will
!    overwrite the input value of B.
!
!  Reference:
!
!    John Dennis, Robert Schnabel,
!    Numerical Methods for Unconstrained Optimization
!    and Nonlinear Equations,
!    SIAM, 1996,
!    ISBN13: 978-0-898713-64-0,
!    LC: QA402.5.D44.
!
!    David Kahaner, Cleve Moler, Steven Nash,
!    Numerical Methods and Software,
!    Prentice Hall, 1989,
!    ISBN: 0-13-627258-4,
!    LC: TA345.K34.
!
!  Parameters:
!
!    Input, integer NR, the leading dimension of A.
!
!    Input, integer N, the number of rows and columns in A.
!
!    Input, real ( kind = 8 ) A(NR,N), the N by N matrix, containing the lower
!    triangular matrix.  A is not altered by this routine.
!
!    Output, real ( kind = 8 ) X(N), the solution vector.
!
!    Input, real ( kind = 8 ) B(N), the right hand side vector.
!
  implicit none

  integer n
  integer nr

  real ( kind = 8 ) a(nr,n)
  real ( kind = 8 ) b(n)
  integer i
  integer ip1
  real ( kind = 8 ) x(n)
!
!  Solve L' * x = b.
!
  i = n
  x(i) = b(i) / a(i,i)

  if ( n == 1 ) then
    return
  end if

  do

    ip1 = i
    i = i - 1

    x(i) = ( b(i) - dot_product ( x(ip1:n), a(ip1:n,i) ) ) / a(i,i)

    if ( i == 1 ) then
      exit
    end if

  end do

  return
end subroutine bakslv
subroutine chlhsn ( nr, n, a, epsm, sx, udiag )

!*****************************************************************************80
!
!! CHLHSN finds the L*L' decomposition of the perturbed model hessian matrix.
!
!  Discussion:
!
!    The perturbed model Hessian matrix has the form
!
!      A + MU * I
!
!    (where 0 <= MU and I is the identity matrix) which is safely
!    positive definite.
!
!    If A is safely positive definite upon entry, then MU=0.
!
!    1. If A has any negative diagonal elements, then choose 0 < MU
!    such that the diagonal of A:=A+MU*I is all positive
!    with the ratio of its smallest to largest element on the
!    order of sqrt ( EPSM ).
!
!    2. A undergoes a perturbed Cholesky decomposition which
!    results in an LL+ decomposition of A+D, where D is a
!    non-negative diagonal matrix which is implicitly added to
!    A during the decomposition if A is not positive definite.
!    A is retained and not changed during this process by
!    copying L into the upper triangular part of A and the
!    diagonal into UDIAG.  Then the Cholesky decomposition routine
!    is called.  On return, ADDMAX contains the maximum element of D.
!
!    3. If ADDMAX=0, A was positive definite going into step 2
!    and return is made to calling program.  Otherwise,
!    the minimum number SDD which must be added to the
!    diagonal of A to make it safely strictly diagonally dominant
!    is calculated.  Since A + ADDMAX * I and A + SDD * I are safely
!    positive definite, choose MU = min ( ADDMAX, SDD ) and decompose
!    A + MU * I to obtain L.
!
!  Reference:
!
!    David Kahaner, Cleve Moler, Steven Nash,
!    Numerical Methods and Software,
!    Prentice Hall, 1989,
!    ISBN: 0-13-627258-4,
!    LC: TA345.K34.
!
!  Parameters:
!
!    Input, integer NR, the row dimension of the matrix.
!
!    Input, integer N, the dimension of the problem.
!
!    Input/output, real A(NR,N), contains an N by N matrix.
!    On input, A is the model hessian.  Only the lower triangular part and
!    diagonal are stored.  On output, A contains the factor L of the
!    LL+ decomposition of the perturbed model hessian in the lower triangular
!    part and diagonal, and contains the hessian in the upper triangular part
!    and UDIAG.
!
!    Input, real ( kind = 8 ) EPSM, the machine epsilon.
!
!    Input, real ( kind = 8 ) SX(N), the diagonal scaling matrix for X.
!
!    Output, real ( kind = 8 ) UDIAG(N), the diagonal of the hessian.
!
!  Local variables:
!
!    tol              tolerance
!    diagmn           minimum element on diagonal of a
!    diagmx           maximum element on diagonal of a
!    offmax           maximum off-diagonal element of a
!    offrow           sum of off-diagonal elements in a row of a
!    evmin            minimum eigenvalue of a
!    evmax            maximum eigenvalue of a
!
  implicit none

  integer n
  integer nr

  real ( kind = 8 ) a(nr,n)
  real ( kind = 8 ) addmax
  real ( kind = 8 ) amu
  real ( kind = 8 ) diagmx
  real ( kind = 8 ) diagmn
  real ( kind = 8 ) epsm
  real ( kind = 8 ) evmax
  real ( kind = 8 ) evmin
  integer i
  integer j
  real ( kind = 8 ) offmax
  real ( kind = 8 ) offrow
  real ( kind = 8 ) posmax
  real ( kind = 8 ) sdd
  real ( kind = 8 ) sx(n)
  real ( kind = 8 ) tol
  real ( kind = 8 ) udiag(n)
!
!  Scale the hessian.
!
  do j = 1, n
    do i = j, n
      a(i,j) = a(i,j) / ( sx(i) * sx(j) )
    end do
  end do
!
!  Step1
!
  tol = sqrt ( epsm )

  diagmx = a(1,1)
  diagmn = a(1,1)

  do i = 2, n
    if ( a(i,i) < diagmn ) then
      diagmn = a(i,i)
    end if
    if ( diagmx < a(i,i) ) then
      diagmx = a(i,i)
    end if
  end do

  posmax = max ( diagmx, 0.0D+00 )

  if ( diagmn <= posmax * tol ) then

    amu = tol * ( posmax - diagmn ) - diagmn
!
!  Find the largest off-diagonal element of A.
!
    if ( amu == 0.0D+00 ) then

      offmax = 0.0D+00

      do i = 2, n
        do j = 1, i-1
          if ( offmax < abs ( a(i,j) ) ) then
            offmax = abs ( a(i,j) )
          end if
        end do
      end do

      amu = offmax

      if ( amu == 0.0D+00 ) then
        amu = 1.0D+00
      else
        amu = amu * ( 1.0D+00 + tol )
      end if

    end if
!
!  A = A + MU*I
!
    do i = 1, n
      a(i,i) = a(i,i) + amu
    end do

    diagmx = diagmx + amu

  end if
!
!  Step2
!
!  Copy lower triangular part of A to upper triangular part
!  and diagonal of A to udiag
!
  do j = 1, n
    udiag(j) = a(j,j)
    do i = j+1, n
      a(j,i) = a(i,j)
    end do
  end do

  call choldc ( nr, n, a, diagmx, tol, addmax )
!
!  Step3
!
!  If ADDMAX=0, A was positive definite going into step 2,
!  the ll+ decomposition has been done, and we return.
!
!  Otherwise, 0 < ADDMAX.  perturb A so that it is safely
!  diagonally dominant and find ll+ decomposition
!
  if ( 0.0D+00 < addmax ) then
!
!  Restore original A (lower triangular part and diagonal)
!
    do j = 1, n
      a(j,j) = udiag(j)
      do i = j+1, n
        a(i,j) = a(j,i)
      end do
    end do
!
!  Find SDD such that A+sdd*i is safely positive definite
!  note:  evmin<0 since A is not positive definite;
!
    evmin = 0.0D+00
    evmax = a(1,1)

    do i = 1, n

      offrow = sum ( abs ( a(i,1:i-1) ) ) + sum ( abs ( a(i+1:n,i) ) )
      evmin = min ( evmin, a(i,i)-offrow )
      evmax = max ( evmax, a(i,i)+offrow )

    end do

    sdd = tol * ( evmax - evmin ) - evmin
!
!  Perturb A and decompose again.
!
    amu = min ( sdd, addmax )

    do i = 1, n
      a(i,i) = a(i,i) + amu
      udiag(i) = a(i,i)
    end do
!
!  A is now guaranteed safely positive definite
!
    call choldc ( nr, n, a, 0.0D+00, tol, addmax )

  end if
!
!  Unscale the hessian and Cholesky decomposition matrix.
!
  do j = 1, n

    a(j:n,j) = sx(j:n) * a(j:n,j)

    do i = 1, j-1
      a(i,j) = sx(i) * sx(j) * a(i,j)
    end do

    udiag(j) = udiag(j) * sx(j) * sx(j)

  end do

  return
end subroutine chlhsn
subroutine choldc ( nr, n, a, diagmx, tol, addmax )

!*****************************************************************************80
!
!! CHOLDC finds the perturbed L*L' decomposition of A+D.
!
!  Discussion:
!
!    D is a non-negative diagonal matrix added to A if
!    necessary to allow the Cholesky decomposition to continue.
!
!    The normal Cholesky decomposition is performed.  However, if at any
!    point the algorithm would attempt to set
!      L(I,I) = sqrt ( TEMP )
!    with
!      TEMP < TOL * DIAGMX,
!    then L(I,I) is set to sqrt ( TOL * DIAGMX )
!    instead.  This is equivalent to adding TOL * DIAGMX-TEMP to A(I,I)
!
!  Reference:
!
!    David Kahaner, Cleve Moler, Steven Nash,
!    Numerical Methods and Software,
!    Prentice Hall, 1989,
!    ISBN: 0-13-627258-4,
!    LC: TA345.K34.
!
!  Parameters:
!
!    Input, integer NR, the row dimension of the matrix.
!
!    Input, integer N, the dimension of the problem.
!
!    Input/output, real ( kind = 8 ) A(NR,N), the N by N matrix.
!    On input, the matrix for which to find the perturbed
!    Cholesky decomposition.
!    On output, the lower triangular part contains the L factor,
!    and the diagonal of A.
!
!    Input, real ( kind = 8 ) DIAGMX, the maximum diagonal element of A.
!
!    Input, real ( kind = 8 ) TOL, a tolerance.
!
!    Output, real ( kind = 8 ) ADDMAX, the maximum amount implicitly added to
!    the diagonal of A in forming the Cholesky decomposition of A+D.
!
!  Local variables:
!
!    aminl    smallest element allowed on diagonal of L.
!
!    amnlsq   =aminl**2
!
!    offmax   maximum off-diagonal element in column of a
!
  implicit none

  integer n
  integer nr

  real ( kind = 8 ) a(nr,n)
  real ( kind = 8 ) addmax
  real ( kind = 8 ) aminl
  real ( kind = 8 ) amnlsq
  real ( kind = 8 ) diagmx
  integer i
  integer j
  integer k
  real ( kind = 8 ) offmax
  real ( kind = 8 ) sum2
  real ( kind = 8 ) temp
  real ( kind = 8 ) tol

  addmax = 0.0D+00
  aminl = sqrt ( diagmx * tol )
  amnlsq = aminl**2
!
!  Form column J of L.
!
  do j = 1, n
!
!  Find diagonal elements of L.
!
    sum2 = sum ( a(j,1:j-1)**2 )

    temp = a(j,j) - sum2

    if ( amnlsq <= temp ) then

      a(j,j) = sqrt ( temp )
!
!  Find maximum off-diagonal element in column.
!
    else

      offmax = 0.0D+00

      do i = j+1, n
        if ( offmax < abs ( a(i,j) ) ) then
          offmax = abs ( a(i,j) )
        end if
      end do

      if ( offmax <= amnlsq ) then
        offmax = amnlsq
      end if
!
!  Add to diagonal element to allow Cholesky decomposition to continue
!
      a(j,j) = sqrt ( offmax )
      addmax = max ( addmax, offmax - temp )

    end if
!
!  Find (I,J) element of lower triangular matrix.
!
    do i = j+1, n
      sum2 = 0.0D+00
      do k = 1, j-1
        sum2 = sum2 + a(i,k) * a(j,k)
      end do
      a(i,j) = ( a(i,j) - sum2 ) / a(j,j)
    end do

  end do

  return
end subroutine choldc
subroutine dogdrv ( nr, n, x, f, g, a, p, xpls, fpls, fcn, sx, stepmx, &
  steptl, dlt, iretcd, mxtake, sc, wrk1, wrk2, wrk3, ipr )

!*****************************************************************************80
!
!! DOGDRV finds the next Newton iterate by the double dogleg method.
!
!  Reference:
!
!    John Dennis, Robert Schnabel,
!    Numerical Methods for Unconstrained Optimization
!    and Nonlinear Equations,
!    SIAM, 1996,
!    ISBN13: 978-0-898713-64-0,
!    LC: QA402.5.D44.
!
!    David Kahaner, Cleve Moler, Steven Nash,
!    Numerical Methods and Software,
!    Prentice Hall, 1989,
!    ISBN: 0-13-627258-4,
!    LC: TA345.K34.
!
!  Parameters:
!
!    Input, integer NR, the row dimension of the matrix.
!
!    Input, integer N, the dimension of the problem.
!
!    Input, real ( kind = 8 ) X(N), the old iterate, "X[K-1]".
!
!    Input, real ( kind = 8 ) F, the function value at the old iterate, "F(X)".
!
!    Input, real ( kind = 8 ) G(N), the gradient at the old iterate.
!
!    Input, real ( kind = 8 ) A(N,N), the Cholesky decomposition of the
!    Hessian matrix in lower triangular part and diagonal.
!
!    Input, real ( kind = 8 ) P(N), the Newton step.
!
!    Output, real ( kind = 8 ) XPLS(N), the new iterate "X[K]".
!
!    Output, real ( kind = 8 ) FPLS, the function value at the new iterate,
!    F(XPLS).
!
!    Input, external FCN, the name of the subroutine to evaluate the function,
!    of the form
!
!      subroutine fcn ( n, x, f )
!      integer n
!      real x(n)
!      real f
!
!    Input, real ( kind = 8 ) SX(N), the diagonal scaling matrix for X.
!
!    Input, real ( kind = 8 ) STEPMX, the maximum allowable step size.
!
!    Input, real ( kind = 8 ) STEPTL, the relative step size at which
!    successive iterates are considered close enough to terminate algorithm.
!
!    Input/output, real ( kind = 8 ) DLT, the trust region radius.
!    [retain value between successive calls].
!
!    Output, integer IRETCD, the return code.
!    0, satisfactory XPLS found
!    1, failed to find satisfactory XPLS sufficiently distinct from X.
!
!    Output, logical MXTAKE, TRUE if a step of maximum length was used.
!
!    Workspace, real ( kind = 8 ) SC(N), holds the current step.
!
!    Workspace, real ( kind = 8 ) WRK1(N).
!
!    Workspace, real ( kind = 8 ) WRK2(N).
!
!    Workspace, real ( kind = 8 ) WRK3(N).
!
!    Input, integer IPR, the device to which to send output.
!
  implicit none

  integer n
  integer nr

  real ( kind = 8 ) a(nr,n)
  real ( kind = 8 ) cln
  real ( kind = 8 ) dlt
  real ( kind = 8 ) eta
  real ( kind = 8 ) f
  external fcn
  real ( kind = 8 ) fpls
  real ( kind = 8 ) fplsp
  logical fstdog
  real ( kind = 8 ) g(n)
  integer ipr
  integer iretcd
  logical mxtake
  logical nwtake
  real ( kind = 8 ) p(n)
  real ( kind = 8 ) rnwtln
  real ( kind = 8 ) sc(n)
  real ( kind = 8 ) stepmx
  real ( kind = 8 ) steptl
  real ( kind = 8 ) sx(n)
  real ( kind = 8 ) wrk1(n)
  real ( kind = 8 ) wrk2(n)
  real ( kind = 8 ) wrk3(n)
  real ( kind = 8 ) x(n)
  real ( kind = 8 ) xpls(n)

  iretcd = 4
  fstdog = .true.

  rnwtln = sqrt ( sum ( sx(1:n)**2 * p(1:n)**2 ) )

  do
!
!  Find new step by double dogleg algorithm.
!
    call dogstp ( nr, n, g, a, p, sx, rnwtln, dlt, nwtake, fstdog, wrk1, &
      wrk2, cln, eta, sc, ipr, stepmx )
!
!  Check new point and update trust region.
!
    call tregup ( nr, n, x, f, g, a, fcn, sc, sx, nwtake, stepmx, steptl, dlt, &
      iretcd, wrk3, fplsp, xpls, fpls, mxtake, ipr, 2, wrk1 )

    if ( iretcd <= 1 ) then
      exit
    end if

  end do

  return
end subroutine dogdrv
subroutine dogstp ( nr, n, g, a, p, sx, rnwtln, dlt, nwtake, fstdog, ssd, v, &
  cln, eta, sc, ipr, stepmx )

!*****************************************************************************80
!
!! DOGSTP finds a new step by the double dogleg algorithm.
!
!  Reference:
!
!    David Kahaner, Cleve Moler, Steven Nash,
!    Numerical Methods and Software,
!    Prentice Hall, 1989,
!    ISBN: 0-13-627258-4,
!    LC: TA345.K34.
!
!  Parameters:
!
!    Input, integer NR, the row dimension of the matrix.
!
!    Input, integer N, the dimension of the problem.
!
!    Input, real ( kind = 8 ) G(N), the gradient at the current iterate.
!
!    Input, real ( kind = 8 ) A(NR,N), the Cholesky decomposition of the
!    hessian in the lower triangle and diagonal.
!
!    Input, real ( kind = 8 ) P(N), the Newton step.
!
!    Input, real ( kind = 8 ) SX(N), the diagonal scaling matrix for X.
!
!    Input, real ( kind = 8 ) RNWTLN, the Newton step length.
!
!    Input/output, real ( kind = 8 ) DLT, the trust region radius.
!
!    Input/output, logical NWTAKE, TRUE if a Newton step was taken.
!
!    Input/output, logical FSTDOG, TRUE if on first leg of dogleg.
!
!    Input/output, real ( kind = 8 ) SSD(N), workspace [cauchy step to
!    the minimum of the quadratic model in the scaled steepest descent
!    direction] [retain value between successive calls]
!
!    Workspace, real ( kind = 8 ) V(N), workspace  [retain value
!    between successive calls]
!
!    Workspace, real ( kind = 8 ) CLN, the cauchy length.
!    [retain value between successive calls]
!
!    Workspace, real ( kind = 8 ) ETA, [retain value between successive calls]
!
!    Output, real ( kind = 8 ) SC(N), the current step.
!
!    Input, integer IPR, the device to which to send output.
!
!    Input, real ( kind = 8 ) STEPMX, the maximum allowable step size.
!
!  Local variables:
!
!    CLN, the length of cauchy step
!
  implicit none

  integer n
  integer nr

  real ( kind = 8 ) a(nr,n)
  real ( kind = 8 ) alam
  real ( kind = 8 ) alpha
  real ( kind = 8 ) beta
  real ( kind = 8 ) cln
  real ( kind = 8 ) dlt
  real ( kind = 8 ) dot1
  real ( kind = 8 ) dot2
  real ( kind = 8 ) eta
  logical fstdog
  real ( kind = 8 ) g(n)
  integer i
  integer ipr
  integer j
  logical nwtake
  real ( kind = 8 ) p(n)
  real ( kind = 8 ) rnwtln
  real ( kind = 8 ) sc(n)
  real ( kind = 8 ) ssd(n)
  real ( kind = 8 ) stepmx
  real ( kind = 8 ) sx(n)
  real ( kind = 8 ) tmp
  real ( kind = 8 ) v(n)
!
!  Can we take a Newton step?
!
  if ( rnwtln <= dlt ) then

    nwtake = .true.
    sc(1:n) = p(1:n)
    dlt = rnwtln

  else
!
!  The Newton step is too long.
!  The Cauchy step is on double dogleg curve.
!
    nwtake = .false.

    if ( fstdog ) then
!
!  Calculate double dogleg curve, SSD.
!
      fstdog = .false.
      alpha = sum ( ( g(1:n) / sx(1:n) )**2 )
      beta = 0.0D+00
      do i = 1, n
        tmp = 0.0D+00
        do j = i, n
          tmp = tmp + ( a(j,i) * g(j) ) / ( sx(j) * sx(j) )
        end do
        beta = beta + tmp * tmp
      end do

      ssd(1:n) = - ( alpha / beta ) * g(1:n) / sx(1:n)

      cln = alpha * sqrt ( alpha ) / beta

      eta = 0.2D+00 + ( 0.8D+00 * alpha * alpha ) / &
        ( - beta * dot_product ( g, p ) )

      v(1:n) = eta * sx(1:n) * p(1:n) - ssd(1:n)

      if ( dlt == - 1.0D+00 ) then
        dlt = min ( cln, stepmx )
      end if

    end if
!
!  Take a partial step in the Newton direction.
!
      if ( eta * rnwtln <= dlt ) then

        sc(1:n) = ( dlt / rnwtln ) * p(1:n)
!
!  Take a step in steepest descent direction.
!
      else if ( dlt <= cln ) then

        sc(1:n) = ( dlt / cln ) * ssd(1:n) / sx(1:n)
!
!  Convex combination of SSD and eta*p which has scaled length DLT.
!
      else

        dot1 = dot_product ( v, ssd )
        dot2 = dot_product ( v, v )
        alam = ( -dot1 + sqrt ( ( dot1 * dot1 ) &
          - dot2 * ( cln * cln - dlt * dlt ) ) ) / dot2
        sc(1:n) = ( ssd(1:n) + alam * v(1:n) ) / sx(1:n)

      end if

  end if

  return
end subroutine dogstp
subroutine forslv ( nr, n, a, x, b )

!*****************************************************************************80
!
!! FORSLV solves A*x=b where A is lower triangular matrix.
!
!  Discussion:
!
!    If B is no longer required by calling routine,
!    then vectors B and X may share the same storage.
!
!  Parameters:
!
!    Input, integer NR, the row dimension of the matrix.
!
!    Input, integer N, the dimension of the problem.
!
!    Input, real ( kind = 8 ) A(NR,N), the N by N lower triangular matrix.
!
!    Output, real ( kind = 8 ) X(N), the solution.
!
!    Input, real ( kind = 8 ) B(N), the right hand side.
!
  implicit none

  integer n
  integer nr

  real ( kind = 8 ) a(nr,n)
  real ( kind = 8 ) b(n)
  integer i
  real ( kind = 8 ) x(n)

  x(1) = b(1) / a(1,1)

  do i = 2, n
    x(i) = ( b(i) - dot_product ( a(i,1:i-1), x(1:i-1) ) ) / a(i,i)
  end do

  return
end subroutine forslv
subroutine fstocd ( n, x, fcn, sx, rnoise, g )

!*****************************************************************************80
!
!! FSTOCD approximates the gradient of a function using central differences.
!
!  Parameters:
!
!    Input, integer N, the dimension of the problem.
!
!    Input, real ( kind = 8 ) X(N), the point at which the gradient is to
!    be approximated.
!
!    Input, external FCN, the name of the subroutine to evaluate the function,
!    of the form
!
!      subroutine fcn ( n, x, f )
!      integer n
!      real x(n)
!      real f
!
!    Input, real ( kind = 8 ) SX(N), the scaling factors for X.
!
!    Input, real ( kind = 8 ) RNOISE, the relative noise in FCN [F(X)].
!
!    Output, real ( kind = 8 ) G(N), a central difference approximation
!    to the gradient.
!
  implicit none

  integer n

  external fcn
  real ( kind = 8 ) fminus
  real ( kind = 8 ) fplus
  real ( kind = 8 ) g(n)
  integer i
  real ( kind = 8 ) rnoise
  real ( kind = 8 ) stepi
  real ( kind = 8 ) sx(n)
  real ( kind = 8 ) third
  real ( kind = 8 ) x(n)
  real ( kind = 8 ) xtempi
!
!  Find I-th stepsize, evaluate two neighbors in direction of I-th
!  unit vector, and evaluate I-th component of gradient.
!
  third = 1.0D+00 / 3.0D+00

  do i = 1, n
    stepi = rnoise**third * max ( abs ( x(i) ), 1.0D+00 / sx(i) )
    xtempi = x(i)
    x(i) = xtempi + stepi
    call fcn ( n, x, fplus )
    x(i) = xtempi - stepi
    call fcn ( n, x, fminus )
    x(i) = xtempi
    g(i) = ( fplus - fminus ) / ( 2.0D+00 * stepi )
  end do

  return
end subroutine fstocd
subroutine fstofd ( nr, m, n, xpls, fcn, fpls, a, sx, rnoise, fhat, icase )

!*****************************************************************************80
!
!! FSTOFD approximates a derivative by a first order approximation.
!
!  Discussion:
!
!    The routine finds the first order forward finite difference
!    approximation A to the first derivative of the function defined
!    by the subprogram "fname" evaluated at the new iterate "xpls".
!
!    For optimization, use this routine to estimate:
!
!    * the first derivative (gradient) of the optimization function "fcn"
!      if no analytic user routine has been supplied;
!
!    * the second derivative (hessian) of the optimization function
!      if no analytic user routine has been supplied for the hessian but
!      one has been supplied for the gradient ("fcn") and if the
!      optimization function is inexpensive to evaluate.
!
!    m=1 (optimization) algorithm estimates the gradient of the function
!    (fcn).   fcn(x) # f: r(n)-->r(1)
!
!    m=n (systems) algorithm estimates the jacobian of the function
!    fcn(x) # f: r(n)-->r(n).
!
!    m=n (optimization) algorithm estimates the hessian of the optimization
!    function, where the hessian is the first derivative of "fcn"
!
!  Reference:
!
!    John Dennis, Robert Schnabel,
!    Numerical Methods for Unconstrained Optimization
!    and Nonlinear Equations,
!    SIAM, 1996,
!    ISBN13: 978-0-898713-64-0,
!    LC: QA402.5.D44.
!
!  Parameters:
!
!    Input, integer NR, the row dimension of the matrix A.
!
!    Input, integer M, the number of rows in A.
!
!    Input, integer N, the number of columns in A, and the dimension
!    of the problem.
!
!    Input, real ( kind = 8 ) XPLS(N), the point at which the derivative is
!    to be estimated.
!
!    Input, external FCN, the name of the subroutine to evaluate
!    the function, of the form
!
!      subroutine fcn ( n, x, f )
!      integer n
!      real x(n)
!      real f
!
!    Input, real ( kind = 8 ) FPLS(M).
!    If M is 1, (optimization), then this is the function value at
!    the new iterate.
!    If M = N for optimization, then this is the value of the first
!    derivative of the function.
!    If M = N for nonlinear systems, then this is the value
!    of the associated minimization function.
!
!    Output, real ( kind = 8 ) A(NR,N), the N by N finite difference
!    approximation.  Only the lower triangular matrix and diagonal are
!    computed.
!
!    Input, real ( kind = 8 ) SX(N), the diagonal scaling matrix for X.
!
!    Input, real ( kind = 8 ) RNOISE, the relative noise or inaccuracy in the
!    function value FCN.
!
!    Workspace, real ( kind = 8 ) FHAT(M).
!
!    Input, integer ICASE, problem specifier:
!    1, optimization (gradient)
!    2, systems
!    3, optimization (hessian)
!
!  Local variables:
!
!    real STEPSZ, the stepsize in the J-th variable direction
!
  implicit none

  integer m
  integer n
  integer nr

  real ( kind = 8 ) a(nr,n)
  external fcn
  real ( kind = 8 ) fhat(m)
  real ( kind = 8 ) fpls(m)
  integer i
  integer icase
  integer j
  real ( kind = 8 ) rnoise
  real ( kind = 8 ) stepsz
  real ( kind = 8 ) sx(n)
  real ( kind = 8 ) xpls(n)
  real ( kind = 8 ) xtmpj
!
!  Find the J-th column of A.
!  Each column is the derivative of f(fcn) with respect to xpls(j).
!
  do j = 1, n
    stepsz = sqrt ( rnoise ) * max ( abs ( xpls(j) ), 1.0D+00 / sx(j) )
    xtmpj = xpls(j)
    xpls(j) = xtmpj + stepsz
    call fcn ( n, xpls, fhat )
    xpls(j) = xtmpj
    a(1:m,j) = ( fhat(1:m) - fpls(1:m) ) / stepsz
  end do

  if ( icase /= 3 ) then
    return
  end if
!
!  If computing the hessian, A must be symmetric.
!
  do j = 1, n-1
    do i = j+1, m
      a(i,j) = ( a(i,j) + a(j,i) ) / 2.0D+00
    end do
  end do

  return
end subroutine fstofd
subroutine grdchk ( n, x, fcn, f, g, typsiz, sx, fscale, rnf, analtl, &
  wrk1, msg, ipr )

!*****************************************************************************80
!
!! GRDCHK checks an analytic gradient against an estimated gradient.
!
!  Reference:
!
!    John Dennis, Robert Schnabel,
!    Numerical Methods for Unconstrained Optimization
!    and Nonlinear Equations,
!    SIAM, 1996,
!    ISBN13: 978-0-898713-64-0,
!    LC: QA402.5.D44.
!
!    David Kahaner, Cleve Moler, Steven Nash,
!    Numerical Methods and Software,
!    Prentice Hall, 1989,
!    ISBN: 0-13-627258-4,
!    LC: TA345.K34.
!
!  Parameters:
!
!    Input, integer N, the dimension of the problem.
!
!    Input, real ( kind = 8 ) X(N), a point at which the gradient is
!    to be checked.
!
!    Input, external FCN, the name of the subroutine that evaluates
!    the optimization function, of the form
!
!      subroutine fcn ( n, x, f )
!      integer n
!      real f
!      real x(n)
!
!    Input, real ( kind = 8 ) F, the function value at X.
!
!    Input, real ( kind = 8 ) G(N), the gradient value at X.
!
!    Input, real ( kind = 8 ) TYPSIZ(N), a typical size for each component of X.
!
!    Input, real ( kind = 8 ) SX(N), the diagonal scaling values:
!    SX(1:N)=1.0/TYPSIZ(1:N)
!
!    Input, real ( kind = 8 ) FSCALE, an estimate of the scale of the
!    objective function FCN.
!
!    Input, real ( kind = 8 ) RNF, the relative noise in the optimization
!    function FCN.
!
!    Input, real ( kind = 8 ) ANALTL, a tolerance for comparison of estimated
!    and analytical gradients
!
!    Workspace, real ( kind = 8 ) WRK1(N).
!
!    Output, integer MSG, message or error code.
!      0: no error detected.
!    -21: probable coding error of gradient
!
!    Input, integer IPR, the device to which to send output.
!
  implicit none

  integer n

  real ( kind = 8 ) analtl
  real ( kind = 8 ) f
  external fcn
  real ( kind = 8 ) fscale
  real ( kind = 8 ) g(n)
  real ( kind = 8 ) gs
  integer i
  integer ipr
  integer ker
  integer msg
  real ( kind = 8 ) rnf
  real ( kind = 8 ) sx(n)
  real ( kind = 8 ) typsiz(n)
  real ( kind = 8 ) value(1)
  real ( kind = 8 ) wrk(1)
  real ( kind = 8 ) wrk1(n)
  real ( kind = 8 ) x(n)

  msg = 0
!
!  Compute the first order finite difference gradient;
!  compare it to the analytic gradient.
!
  value(1) = f
  call fstofd ( 1, 1, n, x, fcn, value, wrk1, sx, rnf, wrk, 1 )

  ker = 0

  do i = 1, n

    gs = max ( abs ( f ), fscale ) / max ( abs ( x(i) ), typsiz(i) )

    if ( max ( abs ( g(i) ), gs ) * analtl < abs ( g(i) - wrk1(i) ) ) then
      ker = 1
    end if

  end do

  if ( ker /= 0 ) then
    write ( ipr, * ) ' '
    write ( ipr, * ) 'GRDCHK - probable error in analytic gradient.'
    write ( ipr, * ) ' '
    write ( ipr, * ) ' grdchk     comp            analytic            est'
    write ( ipr, 902 ) ( i, g(i), wrk1(i), i = 1, n )
    msg = -21
  end if

  return

  902 format(' grdchk    ',i5,3x,e20.13,3x,e20.13)
end subroutine grdchk
subroutine heschk ( nr, n, x, fcn, d1fcn, d2fcn, f, g, a, typsiz, sx, rnf, &
  analtl, iagflg, udiag, wrk1, wrk2, msg, ipr )

!*****************************************************************************80
!
!! HESCHK checks an analytic hessian against a computed estimate.
!
!  Reference:
!
!    John Dennis, Robert Schnabel,
!    Numerical Methods for Unconstrained Optimization
!    and Nonlinear Equations,
!    SIAM, 1996,
!    ISBN13: 978-0-898713-64-0,
!    LC: QA402.5.D44.
!
!    David Kahaner, Cleve Moler, Steven Nash,
!    Numerical Methods and Software,
!    Prentice Hall, 1989,
!    ISBN: 0-13-627258-4,
!    LC: TA345.K34.
!
!  Parameters:
!
!    Input, integer NR, the row dimension of the matrix.
!
!    Input, integer N, the dimension of the problem.
!
!    Input, real ( kind = 8 ) X(N), a point at which the Hessian is to
!    be checked.
!
!    Input, external FCN, the name of the subroutine that evaluates
!    the optimization function, of the form:
!
!      subroutine fcn ( n, x, f )
!      integer n
!      real ( kind = 8 ) f
!      real ( kind = 8 ) x(n)
!
!    Input, external D1FCN, the name of the subroutine to evaluate the
!    gradient of the function, of the form:
!
!      subroutine d1fcn ( n, x, g )
!      integer n
!      real ( kind = 8 ) g(n)
!      real ( kind = 8 ) x(n)
!
!    Input, external D2FCN, the name of the subroutine to evaluate the
!    Hessian of the function, of the form:
!
!      subroutine d2fcn ( nr, n, x, h )
!      integer nr
!      integer n
!      real ( kind = 8 ) h(nr,n)
!      real ( kind = 8 ) x(n)
!
!    Input, real ( kind = 8 ) F, the function value at X.
!
!    Output, real ( kind = 8 ) G(N), the value of the gradient at X.
!
!    Output, real ( kind = 8 ) A(NR,N), the analytic Hessian matrix will
!    be stored in the lower triangle and diagonal.
!
!    Input, real ( kind = 8 ) TYPSIZ(N), a typical size for each component of X.
!
!    Input, real ( kind = 8 ) SX(N), the diagonal scaling matrix.
!
!    Input, real ( kind = 8 ) RNF, the relative noise in the optimization
!    function FCN.
!
!    Input, real ( kind = 8 ) ANALTL, a tolerance for comparison of the
!    estimated and analytic gradients.
!
!    Input, integer IAGFLG, is 1 if the analytic gradient is supplied.
!
!    Workspace, real ( kind = 8 ) UDIAG(N).
!
!    Workspace, real ( kind = 8 ) WRK1(N).
!
!    Workspace, real ( kind = 8 ) WRK2(N).
!
!    Input/output, integer MSG, message or error code
!    on input : if =1xx do not compare analytic + estimated hessian.
!    on output: =-22, probable coding error of hessian.
!
!    Input, integer IPR, the device to which to send output.
!
  implicit none

  integer n
  integer nr

  real ( kind = 8 ) a(nr,n)
  real ( kind = 8 ) analtl
  external d1fcn
  external d2fcn
  real ( kind = 8 ) f
  external fcn
  real ( kind = 8 ) g(n)
  real ( kind = 8 ) hs
  integer i
  integer iagflg
  integer ipr
  integer j
  integer ker
  integer msg
  real ( kind = 8 ) rnf
  real ( kind = 8 ) sx(n)
  real ( kind = 8 ) typsiz(n)
  real ( kind = 8 ) udiag(n)
  real ( kind = 8 ) wrk1(n)
  real ( kind = 8 ) wrk2(n)
  real ( kind = 8 ) x(n)
!
!  Compute the finite difference approximation A to the hessian.
!
  if ( iagflg == 1 ) then
    call fstofd ( nr, n, n, x, d1fcn, g, a, sx, rnf, wrk1, 3 )
  else
    call sndofd ( nr, n, x, fcn, f, a, sx, rnf, wrk1, wrk2 )
  end if

  ker = 0
!
!  Copy lower triangular part of A to upper triangular part
!  and diagonal of A to UDIAG.
!
  do j = 1, n
    udiag(j) = a(j,j)
    do i = j+1, n
      a(j,i) = a(i,j)
    end do
  end do
!
!  Compute analytic hessian and compare to finite difference approximation.
!
  call d2fcn ( nr, n, x, a )

  do j = 1, n

    hs = max ( abs ( g(j) ), 1.0D+00 ) / max ( abs ( x(j) ), typsiz(j) )

    if ( max ( abs ( udiag(j) ), hs ) * analtl &
       < abs ( a(j,j) - udiag(j) ) ) then
      ker = 1
    end if

    do i = j+1, n
      if ( max ( abs ( a(i,j) ), hs ) * analtl &
        < abs ( a(i,j) - a(j,i) ) ) then
        ker = 1
      end if
    end do

  end do

  if ( ker /= 0 ) then

    write ( ipr, '(a)' ) ' '
    write ( ipr, '(a)' ) 'HESCHK:'
    write ( ipr, '(a)' ) '  Probable error in coding of analytic hessian.'
    write ( ipr, '(a)' ) '            row  col              analytic' &
      // '              (estimate)'
    write ( ipr, '(a)' ) ' '

    do i = 1, n
      do j = 1, i-1
        write(ipr,902) i, j, a(i,j), a(j,i)
      end do
      write(ipr,902) i, i, a(i,i), udiag(i)
    end do

    msg = -22

  end if

  return
  902 format('heschk    ',2i5,2x,e20.13,2x,'(',e20.13,')')
end subroutine heschk
subroutine hookdr ( nr, n, x, f, g, a, udiag, p, xpls, fpls, fcn, sx, stepmx, &
  steptl, dlt, iretcd, mxtake, amu, dltp, phi, phip0, sc, xplsp, wrk0, epsm, &
  itncnt, ipr )

!*****************************************************************************80
!
!! HOOKDR finds the next Newton iterate by the More-Hebdon method.
!
!  Reference:
!
!    John Dennis, Robert Schnabel,
!    Numerical Methods for Unconstrained Optimization
!    and Nonlinear Equations,
!    SIAM, 1996,
!    ISBN13: 978-0-898713-64-0,
!    LC: QA402.5.D44.
!
!    David Kahaner, Cleve Moler, Steven Nash,
!    Numerical Methods and Software,
!    Prentice Hall, 1989,
!    ISBN: 0-13-627258-4,
!    LC: TA345.K34.
!
!  Parameters:
!
!    Input, integer NR, the row dimension of the matrix.
!
!    Input, integer N, the dimension of the problem.
!
!    Input, real ( kind = 8 ) X(N), the old iterate, "X[K-1]".
!
!    Input, real ( kind = 8 ) F, the function value at the old iterate.
!
!    Input, real ( kind = 8 ) G(N), the gradient or an approximation, at
!    the old iterate.
!
!    Input, real ( kind = 8 ) A(NR,N), the Cholesky decomposition of hessian
!    in lower triangular part and diagonal.  Hessian in upper triangular
!    part and UDIAG.
!
!    Input, real ( kind = 8 ) UDIAG(N), the diagonal of the hessian matrix.
!
!    Input, real ( kind = 8 ) P(N), the Newton step.
!
!    Output, real ( kind = 8 ) XPLS(N), the new iterate X[K].
!
!    Output, real ( kind = 8 ) FPLS, the function value at the new iterate.
!
!    Input, external FCN, the name of the subroutine to evaluate the function,
!    of the form
!
!      subroutine fcn ( n, x, f )
!      integer n
!      real ( kind = 8 ) x(n)
!      real ( kind = 8 ) f
!
!    Input, real ( kind = 8 ) SX(N), the diagonal scaling matrix for X.
!
!    Input, real ( kind = 8 ) STEPMX, the maximum allowable step size.
!
!    Input, real ( kind = 8 ) STEPTL, the relative step size at which successive
!    iterates are considered close enough to terminate algorithm.
!
!    Input/output, real ( kind = 8 ) DLT, the trust region radius.
!
!    Output, integer IRETCD, return code
!    0, satisfactory xpls found
!    1, failed to find satisfactory xpls sufficiently distinct from x.
!
!    Output, logical MXTAKE, is TRUE if a step of maximum length was used.
!
!    Workspace, real ( kind = 8 ) AMU, [retain value between successive calls]
!
!    Workspace, real ( kind = 8 ) DLTP, [retain value between successive calls]
!
!    Workspace, real ( kind = 8 ) PHI, [retain value between successive calls]
!
!    Workspace, real ( kind = 8 ) PHIP0, [retain value between successive calls]
!
!    Workspace, real ( kind = 8 ) SC(N).
!
!    Workspace, real ( kind = 8 ) XPLSP(N).
!
!    Workspace, real ( kind = 8 ) WRK0(N).
!
!    Input, real ( kind = 8 ) EPSM, the machine epsilon
!
!    Input, integer ITNCNT, the iteration count.
!
!    Input, integer IPR, the device to which to send output.
!
  implicit none

  integer n
  integer nr

  real ( kind = 8 ) a(nr,n)
  real ( kind = 8 ) alpha
  real ( kind = 8 ) amu
  real ( kind = 8 ) beta
  real ( kind = 8 ) dlt
  real ( kind = 8 ) dltp
  real ( kind = 8 ) epsm
  real ( kind = 8 ) f
  external fcn
  real ( kind = 8 ) fpls
  real ( kind = 8 ) fplsp
  logical fstime
  real ( kind = 8 ) g(n)
  integer i
  integer ipr
  integer iretcd
  integer itncnt
  integer j
  logical mxtake
  logical nwtake
  real ( kind = 8 ) p(n)
  real ( kind = 8 ) phi
  real ( kind = 8 ) phip0
  real ( kind = 8 ) rnwtln
  real ( kind = 8 ) sc(n)
  real ( kind = 8 ) stepmx
  real ( kind = 8 ) steptl
  real ( kind = 8 ) sx(n)
  real ( kind = 8 ) tmp
  real ( kind = 8 ) udiag(n)
  real ( kind = 8 ) x(n)
  real ( kind = 8 ) xpls(n)
  real ( kind = 8 ) xplsp(n)
  real ( kind = 8 ) wrk0(n)

  iretcd = 4
  fstime = .true.

  rnwtln = sqrt ( sum ( sx(1:n)**2 * p(1:n)**2 ) )
!
!  If first iteration and trust region not provided by user,
!  compute initial trust region.
!
  if ( itncnt <= 1 ) then

    amu = 0.0D+00

    if ( dlt == -1.0D+00 ) then

      alpha = sum ( ( g(1:n) / sx(1:n) )**2 )

      beta = 0.0D+00
      do i = 1, n
        tmp = 0.0D+00
        do j = i, n
          tmp = tmp + ( a(j,i) * g(j) ) / ( sx(j) * sx(j) )
        end do
        beta = beta + tmp * tmp
      end do

      dlt = alpha * sqrt ( alpha ) / beta
      dlt = min ( dlt, stepmx )

    end if

  end if
!
!  Find the new step by More-Hebdon algorithm.
!
  do

    call hookst ( nr, n, g, a, udiag, p, sx, rnwtln, dlt, amu, dltp, phi, &
      phip0, fstime, sc, nwtake, wrk0, epsm, ipr )

    dltp = dlt
!
!  Check the new point and update trust region.
!
    call tregup ( nr, n, x, f, g, a, fcn, sc, sx, nwtake, stepmx, steptl, &
      dlt, iretcd, xplsp, fplsp, xpls, fpls, mxtake, ipr, 3, udiag )

    if ( iretcd <= 1 ) then
      exit
    end if

  end do

  return
end subroutine hookdr
subroutine hookst ( nr, n, g, a, udiag, p, sx, rnwtln, dlt, amu, &
  dltp, phi, phip0, fstime, sc, nwtake, wrk0, epsm, ipr )

!*****************************************************************************80
!
!! HOOKST finds the new step by the More-Hebdon algorithm.
!
!  Modified:
!
!    15 May 2005
!
!  Reference:
!
!    John Dennis, Robert Schnabel,
!    Numerical Methods for Unconstrained Optimization
!    and Nonlinear Equations,
!    SIAM, 1996,
!    ISBN13: 978-0-898713-64-0,
!    LC: QA402.5.D44.
!
!    David Kahaner, Cleve Moler, Steven Nash,
!    Numerical Methods and Software,
!    Prentice Hall, 1989,
!    ISBN: 0-13-627258-4,
!    LC: TA345.K34.
!
!  Parameters:
!
!    Input, integer NR, the row dimension of matrix.
!
!    Input, integer N, the dimension of the problem.
!
!    Input, real ( kind = 8 ) G(N), the gradient at the current iterate.
!
!    Input, real ( kind = 8 ) A(NR,N), an N by N array.  It contains the
!    Cholesky decomposition of the hessian in the lower triangular
!    part and diagonal; the hessian or approximation in the upper
!    triangular part.
!
!    Input, real ( kind = 8 ) UDIAG(N), the diagonal of the hessian. whose lower
!    triangular part is stored in A.
!
!    Input, real ( kind = 8 ) P(N), the Newton step.
!
!    Input, real ( kind = 8 ) SX(N), the diagonal scaling matrix for X.
!
!    Input, real ( kind = 8 ) RNWTLN, the Newton step length.
!
!    Input/output, real ( kind = 8 ) DLT, the trust region radius.
!
!    Workspace, real ( kind = 8 ) AMU, [retain value between successive calls]
!
!    Input, real ( kind = 8 ) DLTP, the trust region radius at last exit
!    from this routine.
!
!    Workspace, real ( kind = 8 ) PHI, [retain value between successive calls]
!
!    Workspace, real ( kind = 8 ) PHIP0, [retain value between successive calls]
!
!    Input/output, logical FSTIME, TRUE if first entry to this routine
!    during k-th iteration.
!
!    Output, real ( kind = 8 ) SC(N), the current step.
!
!    Output, logical NWTAKE, is TRUE if a Newton step taken.
!
!    Workspace, real ( kind = 8 ) WRK0(N).
!
!    Input, real ( kind = 8 ) EPSM, the machine epsilon.
!
!    Input, integer IPR, the device to which to send output.
!
  implicit none

  integer n
  integer nr

  real ( kind = 8 ) a(nr,n)
  real ( kind = 8 ) addmax
  real ( kind = 8 ), parameter :: alo = 0.75D+00
  real ( kind = 8 ) amu
  real ( kind = 8 ) amulo
  real ( kind = 8 ) amuup
  real ( kind = 8 ) dlt
  real ( kind = 8 ) dltp
  real ( kind = 8 ) dnrm2
  real ( kind = 8 ) epsm
  logical fstime
  real ( kind = 8 ) g(n)
  real ( kind = 8 ), parameter :: hi = 1.50D+00
  integer i
  integer ipr
  integer j
  logical nwtake
  real ( kind = 8 ) p(n)
  real ( kind = 8 ) phi
  real ( kind = 8 ) phip
  real ( kind = 8 ) phip0
  real ( kind = 8 ) rnwtln
  real ( kind = 8 ) sc(n)
  real ( kind = 8 ) stepln
  real ( kind = 8 ) sx(n)
  real ( kind = 8 ) udiag(n)
  real ( kind = 8 ) wrk0(n)
!
!  Take a Newton step?
!
  if ( rnwtln <= hi * dlt ) then
    nwtake = .true.
    sc(1:n) = p(1:n)
    dlt = min ( dlt, rnwtln )
    amu = 0.0D+00
    return
  end if
!
!  Newton step not taken.
!
  nwtake = .false.

  if ( 0.0D+00 < amu ) then
    amu = amu - ( phi + dltp ) * ( ( dltp - dlt ) + phi ) / ( dlt * phip )
  end if

  phi = rnwtln - dlt

  if ( fstime ) then

    wrk0(1:n) = sx(1:n) * sx(1:n) * p(1:n)
!
!  Solve L * Y = (SX**2)*P
!
    call forslv ( nr, n, a, wrk0, wrk0 )

    phip0 = -dnrm2 ( n, wrk0, 1 )**2 / rnwtln
    fstime = .false.

  end if

  phip = phip0
  amulo = -phi / phip
  amuup = 0.0D+00
  do i = 1, n
    amuup = amuup + ( g(i) * g(i) ) / ( sx(i) * sx(i) )
  end do
  amuup = sqrt ( amuup ) / dlt
!
!  Test the value of amu; generate next amu if necessary.
!
  do

    if ( amu < amulo .or. amuup < amu ) then
      amu = max ( sqrt ( amulo * amuup ), amuup * 1.0D-03 )
    end if
!
!  Copy (h,udiag) to L
!  where h <-- h + amu*(sx**2) [do not actually change (h,udiag)]
!
    do j = 1, n
      a(j,j) = udiag(j) + amu * sx(j) * sx(j)
      a(j+1:n,j) = a(j,j+1:n)
    end do
!
!  Factor h=l(l+)
!
    call choldc ( nr, n, a, 0.0D+00, sqrt ( epsm ), addmax )
!
!  Solve h*p = l(l+) * sc = -g
!
    wrk0(1:n) = -g(1:n)

    call lltslv ( nr, n, a, sc, wrk0 )
!
!  Reset H.  Note since UDIAG has not been destroyed, we need do
!  nothing here.  H is in the upper part and in UDIAG, still intact
!
    stepln = sqrt ( dot_product ( sx(1:n)**2, sc(1:n)**2 ) )

    phi = stepln - dlt

    wrk0(1:n) = sx(1:n)**2 * sc(1:n)

    call forslv ( nr, n, a, wrk0, wrk0 )

    phip = -dnrm2 ( n, wrk0, 1 )**2 / stepln
!
!  If SC not acceptable hookstep, then select new AMU.
!
    if ( ( stepln < alo * dlt .or. hi * dlt < stepln ) .and. &
      ( 0.0D+00 < amuup - amulo ) ) then

      amulo = max ( amulo, amu - ( phi / phip ) )

      if ( phi < 0.0D+00 ) then
        amuup = min ( amuup, amu )
      end if

      amu = amu - ( stepln * phi ) / ( dlt * phip )
!
!  SC is acceptable hookstep.
!
    else

      exit

    end if

  end do

  return
end subroutine hookst
subroutine hsnint ( nr, n, a, sx, method )

!*****************************************************************************80
!
!! HSNINT provides initial hessian when using secant updates.
!
!  Reference:
!
!    John Dennis, Robert Schnabel,
!    Numerical Methods for Unconstrained Optimization
!    and Nonlinear Equations,
!    SIAM, 1996,
!    ISBN13: 978-0-898713-64-0,
!    LC: QA402.5.D44.
!
!    David Kahaner, Cleve Moler, Steven Nash,
!    Numerical Methods and Software,
!    Prentice Hall, 1989,
!    ISBN: 0-13-627258-4,
!    LC: TA345.K34.
!
!  Parameters:
!
!    Input, integer NR, the row dimension of the matrix.
!
!    Input, integer N, the dimension of the problem.
!
!    Output, real ( kind = 8 ) A(NR,N), the initial N by N Hessian.  Only the
!    lower triangle of the matrix is assigned values.
!
!    Input, real ( kind = 8 ) SX(N), the scaling factors for X.
!
!    Input, integer METHOD, specifies the algorithm to use to solve
!    the minimization problem.
!    1 or 2: factored secant method used
!    3:  unfactored secant method used
!
  implicit none

  integer n
  integer nr

  real ( kind = 8 ) a(nr,n)
  integer j
  integer method
  real ( kind = 8 ) sx(n)

  do j = 1, n

    if ( method == 3 ) then
      a(j,j) = sx(j)**2
    else
      a(j,j) = sx(j)
    end if

    a(j+1:n,j) = 0.0D+00

  end do

  return
end subroutine hsnint
subroutine lltslv ( nr, n, a, x, b )

!*****************************************************************************80
!
!! LLTSLV solves A*x=b where A = L * L'.
!
!  Discussion:
!
!    L is a lower triangular matrix.
!
!  Reference:
!
!    David Kahaner, Cleve Moler, Steven Nash,
!    Numerical Methods and Software,
!    Prentice Hall, 1989,
!    ISBN: 0-13-627258-4,
!    LC: TA345.K34.
!
!  Parameters:
!
!    Input, integer NR, the row dimension of the matrix.
!
!    Input, integer N, the dimension of the problem.
!
!    Input, real ( kind = 8 ) A(NR,N), contains the lower triangular matrix L.
!
!    Output, real X(N), the solution vector.
!
!    Input, real B(N), the right hand side vector.  If B is not required by
!    the calling program, then B and X may share the same storage.
!
  implicit none

  integer n
  integer nr

  real ( kind = 8 ) a(nr,n)
  real ( kind = 8 ) b(n)
  real ( kind = 8 ) x(n)
!
!  Forward solve, result in X.
!
  call forslv ( nr, n, a, x, b )
!
!  Back solve, result in X.
!
  call bakslv ( nr, n, a, x, x )

  return
end subroutine lltslv
subroutine lnsrch ( n, x, f, g, p, xpls, fpls, fcn, mxtake, iretcd, stepmx, &
  steptl, sx, ipr )

!*****************************************************************************80
!
!! LNSRCH finds a next Newton iterate by line search.
!
!  Reference:
!
!    John Dennis, Robert Schnabel,
!    Numerical Methods for Unconstrained Optimization
!    and Nonlinear Equations,
!    SIAM, 1996,
!    ISBN13: 978-0-898713-64-0,
!    LC: QA402.5.D44.
!
!    David Kahaner, Cleve Moler, Steven Nash,
!    Numerical Methods and Software,
!    Prentice Hall, 1989,
!    ISBN: 0-13-627258-4,
!    LC: TA345.K34.
!
!  Parameters:
!
!    Input, integer N, the dimension of the problem.
!
!    Input, real ( kind = 8 ) X(N), the old iterate, sometimes called X[K-1].
!
!    Input, real ( kind = 8 ) F, the function value at the old iterate.
!
!    Input, real ( kind = 8 ) G(N), the gradient at the old iterate, or an
!    approximation to that value.
!
!    Input, real ( kind = 8 ) P(N), the (non-zero) Newton step.
!
!    Output, real ( kind = 8 ) XPLS(N), the new iterate.
!
!    Output, real ( kind = 8 ) FPLS, the function value at the new iterate.
!
!    Input, external FCN, the name of the subroutine to evaluate the function,
!    of the form
!
!      subroutine fcn ( n, x, f )
!      integer n
!      real ( kind = 8 ) x(n)
!      real ( kind = 8 ) f
!
!    Output, integer IRETCD, the return code.
!
!    Output, logical MXTAKE, is TRUE if a step of maximum size was used.
!
!    Input, real ( kind = 8 ) STEPMX, the maximum allowable step size.
!
!    Input, real ( kind = 8 ) STEPTL, the relative step size at which successive
!    iterates are considered close enough to terminate algorithm.
!
!    Input, real ( kind = 8 ) SX(N), the diagonal scaling matrix for X.
!
!    Input, integer IPR, the device to which to send output.
!
!  Local variables:
!
!    sln, the Newton length.
!
!    rln, the relative length of Newton step
!
  implicit none

  integer n

  real ( kind = 8 ) a
  real ( kind = 8 ) almbda
  real ( kind = 8 ) b
  real ( kind = 8 ) disc
  real ( kind = 8 ) f
  external fcn
  real ( kind = 8 ) fpls
  real ( kind = 8 ) g(n)
  integer i
  integer ipr
  integer iretcd
  logical mxtake
  real ( kind = 8 ) p(n)
  real ( kind = 8 ) pfpls
  real ( kind = 8 ) plmbda
  real ( kind = 8 ) rln
  real ( kind = 8 ) rmnlmb
  real ( kind = 8 ) scl
  real ( kind = 8 ) sln
  real ( kind = 8 ) slp
  real ( kind = 8 ) stepmx
  real ( kind = 8 ) steptl
  real ( kind = 8 ) sx(n)
  real ( kind = 8 ) t1
  real ( kind = 8 ) t2
  real ( kind = 8 ) t3
  real ( kind = 8 ) tlmbda
  real ( kind = 8 ) x(n)
  real ( kind = 8 ) xpls(n)

  mxtake = .false.
  iretcd = 2

  sln = sqrt ( sum ( ( sx(1:n) * p(1:n) )**2 ) )
!
!  Newton step longer than maximum allowed.
!
  if ( stepmx < sln ) then
    scl = stepmx / sln
    p(1:n) = p(1:n) * stepmx / sln
    sln = stepmx
  end if

  slp = dot_product ( g, p )

  rln = 0.0D+00
  do i = 1, n
    rln = max ( rln, abs ( p(i) ) / max ( abs ( x(i) ), 1.0D+00 / sx(i) ) )
  end do

  rmnlmb = steptl / rln
  almbda = 1.0D+00
!
!  Check if new iterate satisfactory.  Generate new lambda if necessary.
!
  do

    if ( iretcd < 2 ) then
      exit
    end if

    xpls(1:n) = x(1:n) + almbda * p(1:n)

    call fcn ( n, xpls, fpls )

    if ( f + slp * 0.0001D+00 * almbda < fpls ) then
      go to 130
    end if
!
!  Solution found.
!
    iretcd = 0

    if ( almbda == 1.0D+00 .and. 0.99D+00 * stepmx < sln ) then
      mxtake = .true.
    end if

    cycle
!
!  Solution not (yet) found.
!
130  continue
!
!  No satisfactory XPLS found sufficiently distinct from X.
!
    if ( almbda < rmnlmb ) then
      iretcd = 1
      cycle
    end if
!
!  Calculate new lambda.
!
!  First backtrack: quadratic fit.
!
    if ( almbda == 1.0D+00 ) then
      tlmbda = -slp / ( 2.0D+00 * ( fpls - f - slp ) )
      go to 170
    end if
!
!  All subsequent backtracks: cubic fit.
!
150 continue

    t1 = fpls - f - almbda * slp
    t2 = pfpls - f - plmbda * slp
    t3 = 1.0D+00 / ( almbda - plmbda )
    a = t3 * ( t1 / ( almbda * almbda ) - t2 / ( plmbda * plmbda ) )
    b = t3 * ( t2 *  almbda / ( plmbda * plmbda ) &
      - t1 * plmbda / ( almbda * almbda ) )
    disc = b * b - 3.0D+00 * a * slp

    if ( disc <= b * b ) then
      go to 160
    end if
!
!  Only one positive critical point, must be minimum.
!
    tlmbda = ( - b + sign ( 1.0D+00, a ) * sqrt ( disc ) ) / ( 3.0D+00 * a )
    go to 165
!
!  Both critical points positive, first is minimum.
!
160 continue

    tlmbda = ( -b - sign ( 1.0D+00, a ) * sqrt ( disc ) ) / ( 3.0D+00 * a )

165 continue

    if ( 0.5D+00 * almbda < tlmbda ) then
      tlmbda = 0.5D+00 * almbda
    end if

170 continue

    plmbda = almbda
    pfpls = fpls

    if ( almbda * 0.1D+00 <= tlmbda ) then
      almbda = tlmbda
    else
      almbda = almbda * 0.1D+00
    end if

  end do

  return
end subroutine lnsrch
subroutine mvmltl ( nr, n, a, x, y )

!*****************************************************************************80
!
!! MVMLTL computes y = L * x where L is a lower triangular matrix stored in A.
!
!  Discussion:
!
!    Note that X and Y cannot share storage.
!
!  Modified:
!
!    29 May 2001
!
!  Parameters:
!
!    Input, integer NR, the row dimension of the matrix.
!
!    Input, integer N, the dimension of the problem.
!
!    Input, real ( kind = 8 ) A(NR,N), the N by N lower triangular matrix.
!
!    Input, real ( kind = 8 ) X(N), the vector to be multiplied.
!
!    Output, real ( kind = 8 ) Y(N), the result.
!
  implicit none

  integer n
  integer nr

  real ( kind = 8 ) a(nr,n)
  integer i
  real ( kind = 8 ) x(n)
  real ( kind = 8 ) y(n)

  do i = 1, n
    y(i) = dot_product ( a(i,1:i), x(1:i) )
  end do

  return
end subroutine mvmltl
subroutine mvmlts ( nr, n, a, x, y )

!*****************************************************************************80
!
!! MVMLTS computes y = A * x where A is a symmetric matrix.
!
!  Discussion:
!
!    A is a symmetric N by N matrix stored in its lower triangular part
!    and X and Y are N vectors.
!
!    X and Y cannot share storage.
!
!  Modified:
!
!    25 August 2001
!
!  Reference:
!
!    David Kahaner, Cleve Moler, Steven Nash,
!    Numerical Methods and Software,
!    Prentice Hall, 1989,
!    ISBN: 0-13-627258-4,
!    LC: TA345.K34.
!
!  Parameters:
!
!    Input, integer NR, the row dimension of the matrix.
!
!    Input, integer N, the dimension of the problem.
!
!    Input, real ( kind = 8 ) A(NR,N), the symmetric N by N matrix.  The entries
!    of A are stored in the lower half of the array.
!
!    Input, real ( kind = 8 ) X(N), the vector to be multiplied by A.
!
!    Output, real ( kind = 8 ) Y(N), the result.
!
  implicit none

  integer n
  integer nr

  real ( kind = 8 ) a(nr,n)
  integer i
  real ( kind = 8 ) x(n)
  real ( kind = 8 ) y(n)

  do i = 1, n

    y(i) = dot_product ( a(i,1:i), x(1:i) ) &
         + dot_product ( a(i+1:n,i), x(i+1:n) )
  end do

  return
end subroutine mvmlts
subroutine mvmltu ( nr, n, a, x, y )

!*****************************************************************************80
!
!! MVMLTU computes y = L' * x where L is a lower triangular matrix.
!
!  Discussion:
!
!    Note that X and Y cannot share storage.
!
!  Reference:
!
!    David Kahaner, Cleve Moler, Steven Nash,
!    Numerical Methods and Software,
!    Prentice Hall, 1989,
!    ISBN: 0-13-627258-4,
!    LC: TA345.K34.
!
!  Parameters:
!
!    Input, integer NR, the row dimension of the matrix.
!
!    Input, integer N, the dimension of the problem.
!
!    Input, real ( kind = 8 ) A(NR,N), the N by N lower triangular matrix,
!
!    Input, real ( kind = 8 ) X(N), the matrix to be multiplied.
!
!    Output, real ( kind = 8 ) Y(N), the result vector.
!
  implicit none

  integer n
  integer nr

  real ( kind = 8 ) a(nr,n)
  integer i
  real ( kind = 8 ) x(n)
  real ( kind = 8 ) y(n)

  do i = 1, n
    y(i) = dot_product ( x(i:n), a(i:n,i) )
  end do

  return
end subroutine mvmltu
subroutine optchk ( n, x, typsiz, sx, fscale, gradtl, itnlim, ndigit, epsm, &
  dlt, method, iexp, iagflg, iahflg, stepmx, msg, ipr )

!*****************************************************************************80
!
!! OPTCHK checks the input to the optimization routine.
!
!  Reference:
!
!    John Dennis, Robert Schnabel,
!    Numerical Methods for Unconstrained Optimization
!    and Nonlinear Equations,
!    SIAM, 1996,
!    ISBN13: 978-0-898713-64-0,
!    LC: QA402.5.D44.
!
!  Parameters:
!
!    Input, integer N, the dimension of the problem.
!
!    Input, real ( kind = 8 ) X(N), an approximate solution of the problem.
!
!    Input/output, real ( kind = 8 ) TYPSIZ(N), a typical size for each
!    component of X.  If TYPSIZ(I) is zero, it is reset to 1.
!
!    Input, real ( kind = 8 ) SX(N), the  diagonal scaling matrix for X.
!
!    Input/output, real ( kind = 8 ) FSCALE, an estimate of the scale of
!    the objective function FCN.
!
!    Input, real ( kind = 8 ) GRADTL, the tolerance at which the gradient
!    is considered close enough to zero to terminate the algorithm.
!
!    Input/output, integer ITNLIM, the maximum number of allowable
!    iterations.
!
!    Input/output, integer NDIGIT, the number of good digits in
!    optimization function FCN.
!
!    Input, real ( kind = 8 ) EPSM, the machine epsilon.
!
!    Input/output, real ( kind = 8 ) DLT, the trust region radius.
!
!    Input/output, integer METHOD, the algorithm indicator.
!
!    Input/output, integer IEXP, the expense flag.
!
!    Input/output, integer IAGFLG, = 1 if analytic gradient supplied.
!
!    Input/output, integer IAHFLG, = 1 if analytic hessian supplied.
!
!    Input/output, real ( kind = 8 ) STEPMX, the maximum step size.
!
!    Input/output, integer MSG, the message and error code.
!
!    Input, integer IPR, the device to which to send output.
!
  implicit none

  integer n

  real ( kind = 8 ) dlt
  real ( kind = 8 ) epsm
  real ( kind = 8 ) fscale
  real ( kind = 8 ) gradtl
  integer i
  integer iagflg
  integer iahflg
  integer iexp
  integer ipr
  integer itnlim
  integer method
  integer msg
  integer ndigit
  real ( kind = 8 ) stepmx
  real ( kind = 8 ) stpsiz
  real ( kind = 8 ) sx(n)
  real ( kind = 8 ) typsiz(n)
  real ( kind = 8 ) x(n)
!
!  Check that parameters only take on acceptable values.
!  if not, set them to default values.
!
  if ( method < 1 .or. 3 < method ) then
    method = 1
  end if

  if ( iagflg /= 1 ) then
    iagflg = 0
  end if

  if ( iahflg /= 1 ) then
    iahflg = 0
  end if

  if ( iexp /= 0 ) then
    iexp = 1
  end if

  if ( mod ( msg/2, 2 ) == 1 .and. iagflg == 0 ) then
    write ( ipr, 906 ) msg, iagflg
    msg = -6
    return
  end if

  if ( mod ( msg/4, 2 ) == 1 .and. iahflg == 0 ) then
    write ( ipr, 907 ) msg, iahflg
    msg = -7
    return
  end if
!
!  Check N.
!
  if ( n <= 0 ) then
    write ( ipr, * ) ' '
    write ( ipr, * ) 'OPTCHK - Fatal error!'
    write ( ipr, * ) '  Illegal nonpositive value of N = ', n
    msg = -1
    return
  end if

  if ( n == 1 .and. mod ( msg, 2 ) == 0 ) then
    write ( ipr, 902 )
    msg = -2
    return
  end if
!
!  Compute the scale matrix.
!
  do i = 1, n
    if ( typsiz(i) == 0.0D+00 ) then
      typsiz(i) = 1.0D+00
    end if
  end do

  typsiz(1:n) = abs ( typsiz(1:n) )
  sx(1:n) = 1.0D+00 / typsiz(1:n)
!
!  Check maximum step size.
!
  if ( stepmx <= 0.0D+00 ) then

    stpsiz = sqrt ( sum ( x(1:n)**2 * sx(1:n)**2 ) )

    stepmx = max ( 1.0D+03 * stpsiz, 1.0D+03 )

  end if
!
!  Check the function scale.
!
  if ( fscale == 0.0D+00 ) then
    fscale = 1.0D+00
  end if

  if ( fscale < 0.0D+00 ) then
    fscale = -fscale
  end if
!
!  Check gradient tolerance
!
  if ( gradtl < 0.0D+00 ) then
    write ( ipr, 903 ) gradtl
    msg = -3
    return
  end if
!
!  Check iteration limit
!
  if ( itnlim <= 0 ) then
    write ( ipr, 904 ) itnlim
    msg = -4
    return
  end if
!
!  Check number of digits of accuracy in function FCN.
!
  if ( ndigit == 0 ) then
    write ( ipr, 905 ) ndigit
    msg = -5
    return
  end if

  if ( ndigit < 0 ) then
    ndigit = -log10 ( epsm )
  end if
!
!  Check trust region radius.
!
  if ( dlt <= 0.0D+00 ) then
    dlt = -1.0D+00
  end if

  if ( stepmx < dlt ) then
    dlt = stepmx
  end if

  902 format(' optchk    +++ warning +++  this package is inefficient', &
    'for problems of size n=1.'/ &
    ' optchk    check installation libraries for more appropriate routines.'/ &
    ' optchk    if none, set msg and resubmit.')
  903 format(' optchk    illegal tolerance.  gradtl=',e20.13)
  904 format(' optchk    illegal iteration limit.  itnlim=',i5)
  905 format(' optchk    minimization function has no good digits.', &
     'ndigit=',i5)
  906 format(' optchk    user requests that analytic gradient be', &
     ' accepted as properly coded (msg=',i5, '),'/ &
     ' optchk    but analytic gradient not supplied (iagflg=',i5, ').')
  907 format(' optchk    user requests that analytic hessian be', &
     ' accepted as properly coded (msg=',i5, '),'/ &
     ' optchk    but analytic hessian not supplied(iahflg=',i5, ').')

  return
end subroutine optchk
subroutine optdrv ( nr, n, x, fcn, d1fcn, d2fcn, typsiz, fscale, method, &
  iexp, msg, ndigit, itnlim, iagflg, iahflg, ipr, dlt, gradtl, stepmx, &
  steptl, xpls, fpls, gpls, itrmcd, a, udiag, g, p, sx, wrk0, wrk1, wrk2, &
  wrk3 )

!*****************************************************************************80
!
!! OPTDRV is a driver for the nonlinear optimization package.
!
!  Reference:
!
!    John Dennis, Robert Schnabel,
!    Numerical Methods for Unconstrained Optimization
!    and Nonlinear Equations,
!    SIAM, 1996,
!    ISBN13: 978-0-898713-64-0,
!    LC: QA402.5.D44.
!
!  Parameters:
!
!    Input, integer NR, the row dimension of the matrix.
!
!    Input, integer N, the dimension of the problem.
!
!    Input/output, real ( kind = 8 ) X(N).  On input, an rough solution of
!    the problem.  On output, the computed solution.
!
!    Input, external FCN, the name of the subroutine that evaluates
!    the optimization function, of the form
!
!      subroutine fcn ( n, x, f )
!      integer n
!      real ( kind = 8 ) f
!      real ( kind = 8 ) x(n)
!
!    Input, external D1FCN, the name of the subroutine to evaluate gradient
!    of FCN, of the form
!
!      subroutine d1fcn ( n, x, g )
!      integer n
!      real ( kind = 8 ) g(n)
!      real ( kind = 8 ) x(n)
!
!    Input, external D2FCN, the name of the subroutine to evaluate the
!    Hessian of the function, of the form:
!
!      subroutine d2fcn ( nr, n, x, h )
!      integer nr
!      integer n
!      real ( kind = 8 ) h(nr,n)
!      real ( kind = 8 ) x(n)
!
!    Input, real ( kind = 8 ) TYPSIZ(N), a typical size for each component of X.
!
!    Input, real ( kind = 8 ) FSCALE, an estimate of the scale of the objective
!    function.
!
!    Input, integer METHOD, the algorithm to use to solve minimization problem:
!    1, line search
!    2, double dogleg
!    3, More-Hebdon
!
!    Input, integer IEXP, function expense flag.
!    Set IEXP to 1 if optimization function fcn is expensive to
!    evaluate,  and 0 otherwise.  If set then hessian will
!    be evaluated by secant update instead of
!    analytically or by finite differences.
!
!    Input/output, integer MSG.
!    On input, set it positive to inhibit certain automatic checks
!    On output. < 0 indicates an error occurred.
!
!    Input, integer NDIGIT, the number of good digits in optimization
!    function fcn.
!
!    Input, integer ITNLIM, the maximum number of allowable iterations.
!
!    Input, integer IAGFLG, is 1 if analytic gradient supplied.
!
!    Input, integer IAHFLG, is 1 if analytic hessian supplied.
!
!    Input, integer IPR, the device to which to send output.
!
!    Input, real ( kind = 8 ) DLT, the trust region radius.
!
!    Input, real ( kind = 8 ) GRADTL, the tolerance at which gradient
!    considered close enough to zero to terminate algorithm.
!
!    Input, real ( kind = 8 ) STEPMX, the maximum allowable step size.
!
!    Input, real ( kind = 8 ) STEPTL, the relative step size at which
!    successive iterates are considered close enough to terminate algorithm
!
!    Input/output, real ( kind = 8 ) XPLS(N); on exit, XPLS is the local
!    minimizer.
!
!    Input/output, real ( kind = 8 ) FPLS; on exit, the function value at XPLS.
!
!    Input/output, real ( kind = 8 ) GPLS(N); on exit, the gradient at XPLS.
!
!    Output, integer ITRMCD, the termination code.
!
!    Workspace, real ( kind = 8 ) A(NR,N), workspace for hessian (or estimate)
!    and its Cholesky decomposition.
!
!    Workspace, real ( kind = 8 ) UDIAG(N), workspace for diagonal of hessian.
!
!    Workspace, real ( kind = 8 ) G(N), workspace for gradient at current
!    iterate.
!
!    Workspace, real ( kind = 8 ) P(N), workspace for the step.
!
!    Workspace, real ( kind = 8 ) SX(N), workspace for diagonal scaling matrix.
!
!    Workspace, real ( kind = 8 ) WRK0(N), WRK1(N), WRK2(N), WRK3(N).
!
!  Local variables:
!
!    analtl, tolerance for gradient and hessian checking.
!
!    epsm, machine epsilon.
!
!    f, function value: fcn(x).
!
!    itncnt, current iteration, k
!
!    rnf, relative noise in optimization function fcn.
!
!    noise=10.**(-ndigit)
!
  implicit none

  integer n
  integer nr

  real ( kind = 8 ) a(nr,n)
  real ( kind = 8 ) amu
  real ( kind = 8 ) amusav
  real ( kind = 8 ) analtl
  external d1fcn
  external d2fcn
  real ( kind = 8 ) dlpsav
  real ( kind = 8 ) dlt
  real ( kind = 8 ) dltp
  real ( kind = 8 ) dltsav
  real ( kind = 8 ) epsm
  real ( kind = 8 ) f
  external fcn
  real ( kind = 8 ) fpls
  real ( kind = 8 ) fscale
  real ( kind = 8 ) g(n)
  real ( kind = 8 ) gpls(n)
  real ( kind = 8 ) gradtl
  integer iagflg
  integer iahflg
  integer icscmx
  integer iexp
  integer ipr
  integer iretcd
  integer itncnt
  integer itnlim
  integer itrmcd
  integer method
  integer msg
  logical mxtake
  integer ndigit
  logical noupdt
  real ( kind = 8 ) p(n)
  real ( kind = 8 ) phi
  real ( kind = 8 ) phip0
  real ( kind = 8 ) phisav
  real ( kind = 8 ) phpsav
  real ( kind = 8 ) rnf
  real ( kind = 8 ) stepmx
  real ( kind = 8 ) steptl
  real ( kind = 8 ) sx(n)
  real ( kind = 8 ) typsiz(n)
  real ( kind = 8 ) udiag(n)
  real ( kind = 8 ) value(1)
  real ( kind = 8 ) wrk(1)
  real ( kind = 8 ) wrk0(n)
  real ( kind = 8 ) wrk1(n)
  real ( kind = 8 ) wrk2(n)
  real ( kind = 8 ) wrk3(n)
  real ( kind = 8 ) x(n)
  real ( kind = 8 ) xpls(n)
!
!  Initialization.
!
  p(1:n) = 0.0D+00
  itncnt = 0
  iretcd = -1
  epsm = epsilon ( epsm )

  call optchk ( n, x, typsiz, sx, fscale, gradtl, itnlim, ndigit, epsm, &
    dlt, method, iexp, iagflg, iahflg, stepmx, msg, ipr )

  if ( msg < 0 ) then
    return
  end if

  rnf = max ( 10.0D+00**(-ndigit), epsm )

  analtl = max ( 1.0D-02, sqrt ( rnf ) )

  if ( mod ( msg / 8, 2 ) == 0 ) then
    write ( ipr, 901 )
    write ( ipr, 900 ) typsiz(1:n)
    write ( ipr, 902 )
    write ( ipr, 900 ) sx(1:n)
    write ( ipr, 903 ) fscale
    write ( ipr, 904 ) ndigit,iagflg,iahflg,iexp,method,itnlim,epsm
    write ( ipr, 905 ) stepmx,steptl,gradtl,dlt,rnf,analtl
  end if
!
!  Evaluate fcn(x)
!
  call fcn ( n, x, f )
!
!  Evaluate analytic or finite difference gradient and check analytic
!  gradient, if requested.
!
  if ( iagflg /= 1 ) then

    value(1) = f
    call fstofd ( 1, 1, n, x, fcn, value, g, sx, rnf, wrk, 1 )

  else

    call d1fcn ( n, x, g )

    if ( mod ( msg/2, 2 ) /= 1 ) then

      call grdchk ( n, x, fcn, f, g, typsiz, sx, fscale, rnf, analtl, wrk1, &
        msg, ipr )

      if ( msg < 0 ) then
        return
      end if

    end if

  end if

  call optstp ( n, x, f, g, wrk1, itncnt, icscmx, itrmcd, gradtl, steptl, &
    sx, fscale, itnlim, iretcd, mxtake, ipr, msg )

  if ( itrmcd /= 0 ) then
    go to 700
  end if

  if ( iexp /= 1 ) then
    go to 80
  end if
!
!  If optimization function expensive to evaluate (iexp=1), then
!  hessian will be obtained by secant updates.  Get initial hessian.
!
  call hsnint ( nr, n, a, sx, method )
  go to 90

80 continue
!
!  Evaluate analytic or finite difference hessian and check analytic
!  hessian if requested (only if user-supplied analytic hessian
!  routine d2fcn fills only lower triangular part and diagonal of a).
!
  if ( iahflg == 1 ) then
    go to 82
  end if

  if ( iagflg == 1 ) then
    call fstofd ( nr, n, n, x, d1fcn, g, a, sx, rnf, wrk1, 3 )
  else
    call sndofd ( nr, n, x, fcn, f, a, sx, rnf, wrk1, wrk2 )
  end if

  go to 88

82 continue

  if ( mod ( msg / 4, 2 ) == 0 ) then
    go to 85
  end if

  call d2fcn ( nr, n, x, a )
  go to 88

85 continue

  call heschk ( nr, n, x, fcn, d1fcn, d2fcn, f, g, a, typsiz, &
    sx, rnf, analtl, iagflg, udiag, wrk1, wrk2, msg, ipr )
!
!  HESCHK evaluates d2fcn and checks it against the finite
!  difference hessian which it calculates by calling fstofd
!  (if iagflg == 1) or sndofd (otherwise).
!
  if ( msg < 0 ) then
    return
  end if

88 continue

90 continue

  if ( mod ( msg / 8, 2 ) == 0 ) then
    call result ( nr, n, x, f, g, a, p, itncnt, 1, ipr )
  end if
!
!  iteration
!
  100 continue

  itncnt = itncnt + 1
!
!  Find perturbed local model hessian and its ll+ decomposition
!  (skip this step if line search or dogstep techniques being used with
!  secant updates.  Cholesky decomposition l already obtained from
!  secfac.)
!
  if ( iexp == 1 .and. method /= 3 ) then
    go to 105
  end if

  103   continue

  call chlhsn ( nr, n, a, epsm, sx, udiag )
  105 continue
!
!  Solve for Newton step:  ap = -g
!
  wrk1(1:n) = - g(1:n)

  call lltslv ( nr, n, a, p, wrk1 )
!
!  Decide whether to accept Newton step  xpls = x + p
!  or to choose xpls by a global strategy.
!
  if ( iagflg == 0 .and. method /= 1 ) then

    dltsav = dlt

    if ( method /= 2 ) then
      amusav = amu
      dlpsav = dltp
      phisav = phi
      phpsav = phip0
    end if

  end if

  if ( method == 1 ) then

    call lnsrch ( n, x, f, g, p, xpls, fpls, fcn, mxtake, iretcd, &
      stepmx, steptl, sx, ipr )

  else if ( method == 2 ) then

    call dogdrv ( nr, n, x, f, g, a, p, xpls, fpls, fcn, sx, stepmx, &
      steptl, dlt, iretcd, mxtake, wrk0, wrk1, wrk2, wrk3, ipr )

  else if ( method == 3 ) then

    call hookdr ( nr, n, x, f, g, a, udiag, p, xpls, fpls, fcn, sx, stepmx, &
      steptl, dlt, iretcd, mxtake, amu, dltp, phi, phip0, wrk0, &
      wrk1, wrk2, epsm, itncnt, ipr )

  end if
!
!  If could not find satisfactory step and forward difference
!  gradient was used, retry using central difference gradient.
!
  if ( iretcd /= 1 .or. iagflg /= 0 ) then
    go to 112
  end if
!
!  Set iagflg for central differences.
!
     iagflg = -1
     write(ipr,906) itncnt
     call fstocd ( n, x, fcn, sx, rnf, g )

     if ( method == 1 ) then
       go to 105
     end if

     dlt = dltsav

     if ( method == 2 ) then
       go to 105
     end if

     amu = amusav
     dltp = dlpsav
     phi = phisav
     phip0 = phpsav
     go to 103
!
!  Calculate step for output
!
  112 continue

  p(1:n) = xpls(1:n) - x(1:n)
!
!  Calculate the gradient at XPLS.
!
  if ( iagflg == -1 ) then
    call fstocd ( n, xpls, fcn, sx, rnf, gpls )
  else if ( iagflg == 0 ) then
    value(1) = fpls
    call fstofd ( 1, 1, n, xpls, fcn, value, gpls, sx, rnf, wrk, 1 )
  else
    call d1fcn ( n, xpls, gpls )
  end if
!
!  Check whether stopping criteria satisfied.
!
  call optstp ( n, xpls, fpls, gpls, x, itncnt, icscmx, itrmcd, gradtl, &
    steptl, sx, fscale, itnlim, iretcd, mxtake, ipr, msg )

  if ( itrmcd /= 0 ) then
    go to 690
  end if
!
!  Evaluate hessian at xpls
!
  if ( iexp == 0 ) then
    go to 130
  end if

  if ( method == 3 ) then
     call secunf ( nr, n, x, g, a, udiag, xpls, gpls, epsm, itncnt, rnf, &
       iagflg, noupdt, wrk1, wrk2, wrk3 )
  else
    call secfac ( nr, n, x, g, a, xpls, gpls, epsm, itncnt, rnf, iagflg, &
      noupdt, wrk0, wrk1, wrk2, wrk3 )
  end if

  go to 150

  130 continue

  if ( iahflg == 1 ) then
    go to 140
  end if

  if ( iagflg == 1 ) then
    call fstofd ( nr, n, n, xpls, d1fcn, gpls, a, sx, rnf, wrk1, 3 )
  else
    call sndofd ( nr, n, xpls, fcn, fpls, a, sx, rnf, wrk1, wrk2 )
  end if

  go to 150

  140 continue

  call d2fcn ( nr, n, xpls, a )

  150 continue

  if ( mod ( msg / 16, 2 ) == 1 ) then
    call result ( nr, n, xpls, fpls, gpls, a, p, itncnt, 1, ipr )
  end if
!
!  x <-- xpls
!  g <-- gpls
!  f <-- fpls
!
  f = fpls
  x(1:n) = xpls(1:n)
  g(1:n) = gpls(1:n)

  go to 100
!
!  Termination.
!
!  Reset XPLS, FPLS, GPLS, if previous iterate solution
!
  690 if ( itrmcd /= 3 ) then
    go to 710
  end if

  700 continue

  fpls = f
  xpls(1:n) = x(1:n)
  gpls(1:n) = g(1:n)
!
!  Print results
!
  710 continue

  if ( mod ( msg / 8, 2 ) == 0) then
    call result ( nr, n, xpls, fpls, gpls, a, p, itncnt, 0, ipr )
  end if

  msg = 0

  900 format(' optdrv       ',5(e20.13,3x))
  901 format('0optdrv    typical x')
  902 format(' optdrv    diagonal scaling matrix for x')
  903 format(' optdrv    typical f =',e20.13)
  904 format('0optdrv    number of good digits in fcn=',i5/ &
             ' optdrv    gradient flag  =',i5,'   (=1 if analytic', &
             ' gradient supplied)'/ &
             ' optdrv    hessian flag   =',i5,'   (=1 if analytic', &
             ' hessian supplied)'/ &
             ' optdrv    expense flag   =',i5, '   (=1 if', &
             ' minimization function expensive to evaluate)'/ &
             ' optdrv    method to use  =',i5,'   (=1,2,3 for line', &
             ' search, double dogleg, more-hebdon respectively)'/ &
             ' optdrv    iteration limit=',i5/ &
             ' optdrv    machine epsilon=',e20.13)

  905 format('0optdrv    maximum step size =',e20.13/ &
             ' optdrv    step tolerance    =',e20.13/ &
             ' optdrv    gradient tolerance=',e20.13/ &
             ' optdrv    trust reg radius  =',e20.13/ &
             ' optdrv    rel noise in fcn  =',e20.13/ &
             ' optdrv    anal-fd tolerance =',e20.13)

  906 format(' optdrv    shift from forward to central differences', &
     ' in iteration ', i5)

  return
end subroutine optdrv
subroutine optstp ( n, xpls, fpls, gpls, x, itncnt, icscmx, &
  itrmcd, gradtl, steptl, sx, fscale, itnlim, iretcd, mxtake, ipr, msg )

!*****************************************************************************80
!
!! OPTSTP: unconstrained minimization stopping criteria
!
!  Discussion:
!
!    OPSTP determines whether the optimization algorithm should terminate,
!    due to any of the following:
!    1) the problem has been solved to the user's tolerance;
!    2) convergence within user tolerance;
!    3) iteration limit reached;
!    4) divergence or too restrictive maximum step (stepmx) suspected;
!
!  Reference:
!
!    John Dennis, Robert Schnabel,
!    Numerical Methods for Unconstrained Optimization
!    and Nonlinear Equations,
!    SIAM, 1996,
!    ISBN13: 978-0-898713-64-0,
!    LC: QA402.5.D44.
!
!  Parameters:
!
!    Input, integer N, the dimension of the problem.
!
!    Input, real ( kind = 8 ) XPLS(N), the new iterate, X[K].
!
!    Input, real ( kind = 8 ) FPLS, the function value at the new iterate.
!
!    Input, real ( kind = 8 ) GPLS(N), the gradient at the new iterate, or an
!    approximation of that value.
!
!    Input, real ( kind = 8 ) X(N), the old iterate X[K-1].
!
!    Input, integer ITNCNT, the current iteration K.
!
!    Input/output, integer ICSCMX, the number of consecutive steps
!    greater than or equal to STEPMX.
!    [retain value between successive calls].
!
!    Output, integer ITRMD, the termination code.
!
!    Input, real ( kind = 8 ) GRADTL, the tolerance at which relative gradient
!    considered close enough to zero to terminate algorithm.
!
!    Input, real ( kind = 8 ) STEPTL, the relative step size at which
!    successive iterates are considered close enough to terminate algorithm.
!
!    Input, real ( kind = 8 ) SX(N), the diagonal scaling matrix for X.
!
!    Input, real ( kind = 8 ) FSCALE, an estimate of the scale of objective
!    function.
!
!    Input, integer ITNLIM, the maximum number of allowable iterations.
!
!    Output, integer IRETCD, the return code.
!
!    Input, logical MXTAKE, TRUE if a step of maximum length was used.
!
!    Output, integer IPR, the device to which to send output.
!
!    Input, integer MSG, if includes a term 8, suppress output.
!
  implicit none

  integer n

  real ( kind = 8 ) d
  real ( kind = 8 ) fpls
  real ( kind = 8 ) fscale
  real ( kind = 8 ) gpls(n)
  real ( kind = 8 ) gradtl
  integer i
  integer icscmx
  integer ipr
  integer iretcd
  integer itncnt
  integer itnlim
  integer itrmcd
  integer jtrmcd
  integer msg
  logical mxtake
  real ( kind = 8 ) relgrd
  real ( kind = 8 ) relstp
  real ( kind = 8 ) rgx
  real ( kind = 8 ) rsx
  real ( kind = 8 ) steptl
  real ( kind = 8 ) sx(n)
  real ( kind = 8 ) x(n)
  real ( kind = 8 ) xpls(n)

  itrmcd = 0
!
!  Last global step failed to locate a point lower than X.
!
  if ( iretcd == 1 ) then
    jtrmcd = 3
    go to 600
  end if
!
!  Find direction in which relative gradient maximum.
!  Check whether within tolerance
!
  d = max ( abs ( fpls ), fscale )

  rgx = 0.0D+00
  do i = 1, n
    relgrd = abs ( gpls(i) ) * max ( abs ( xpls(i) ), 1.0D+00 / sx(i) ) / d
    rgx = max ( rgx, relgrd )
  end do

  jtrmcd = 1
  if ( rgx <= gradtl ) then
    go to 600
  end if

  if ( itncnt == 0 ) then
    return
  end if
!
!  Find direction in which relative stepsize is maximum.
!  Check whether within tolerance.
!
  rsx = 0.0D+00
  do i = 1, n
    relstp = abs ( xpls(i) - x(i) ) / max ( abs ( xpls(i) ), 1.0D+00 / sx(i) )
    rsx = max ( rsx, relstp )
  end do

  jtrmcd = 2
  if ( rsx <= steptl ) then
    go to 600
  end if
!
!  Check iteration limit.
!
  jtrmcd = 4
  if ( itnlim <= itncnt ) then
    go to 600
  end if
!
!  Check number of consecutive steps \ stepmx
!
  if ( .not. mxtake ) then
    icscmx = 0
    return
  else
    if ( mod ( msg / 8, 2 ) == 0 ) then
      write ( ipr, 900 )
    end if
    icscmx = icscmx + 1
    if ( icscmx < 5 ) then
      return
    end if
    jtrmcd = 5
  end if
!
!  Print termination code
!
  600 continue

  itrmcd = jtrmcd

  if ( itrmcd == 1 ) then
    write ( ipr, 901 )
  else if ( itrmcd == 2 ) then
    write(ipr,902)
  else if ( itrmcd == 3 ) then
    write(ipr,903)
  else if ( itrmcd == 4 ) then
    write(ipr,904)
  else if ( itrmcd == 5 ) then
    write(ipr,905)
  end if

  900 format('0optstp    step of maximum length (stepmx) taken')
  901 format('0optstp    relative gradient close to zero.'/ &
             ' optstp    current iterate is probably solution.')
  902 format('0optstp    successive iterates within tolerance.'/ &
             ' optstp    current iterate is probably solution')
  903 format('0optstp    last global step failed to locate a point', &
             ' lower than x.'/ &
             ' optstp    either x is an approximate local minimum', &
             ' of the function',/ &
             ' optstp    the function is too non-linear for this algorithm,'/ &
             ' optstp    or steptl is too large.')
  904 format('optstp    iteration limit exceeded.'/'optstp    algorithm failed.')
  905 format('0optstp    maximum step size exceeded 5 consecutive times.'/ &
             ' optstp    either the function is unbounded below',/ &
             ' optstp    becomes asymptotic to a finite value', &
             ' from above in some direction',/ &
             ' optstp    or stepmx is too small')

  return
end subroutine optstp
subroutine qraux1 ( nr, n, r, i )

!*****************************************************************************80
!
!! QRAUX1 interchanges two rows of an upper Hessenberg matrix.
!
!  Discussion:
!
!    QRAUX1 interchanges rows I and I+1 of the upper Hessenberg matrix
!    R, columns I to N.
!
!  Parameters:
!
!    Input, integer NR, the leading dimension of the matrix.
!
!    Input, integer N, the dimension of the matrix.
!
!    Input/output, real ( kind = 8 ) R(NR,N), the N by N upper Hessenberg
!    matrix.
!
!    Input, integer I, the index of the first row to interchange.
!
  implicit none

  integer n
  integer nr

  integer i
  integer j
  real ( kind = 8 ) r(nr,n)

  do j = i, n
    call r8_swap ( r(i,j), r(i+1,j) )
  end do

  return
end subroutine qraux1
subroutine qraux2 ( nr, n, r, i, a, b )

!*****************************************************************************80
!
!! QRAUX2 pre-multiplies an upper Hessenberg matrix by a Jacobi rotation.
!
!  Discussion:
!
!    QRAUX2 pre-multiplies an upper Hessenberg matrix by a Jacobi rotation
!    J(I,I+1,A,B)
!
!  Modified:
!
!    15 December 2001
!
!  Parameters:
!
!    Input, integer NR, the row dimension of the matrix.
!
!    Input, integer N, the order of the matrix.
!
!    Input/output, real ( kind = 8 ) R(NR,N), the N by N upper Hessenberg
!    matrix.
!
!    Input, integer I, the index of the row.
!
!    Input, real ( kind = 8 ) A, B, scalars that define the rotation.
!
  implicit none

  integer n
  integer nr

  real ( kind = 8 ) a
  real ( kind = 8 ) b
  real ( kind = 8 ) c
  real ( kind = 8 ) den
  integer i
  integer j
  real ( kind = 8 ) r(nr,n)
  real ( kind = 8 ) s
  real ( kind = 8 ) y
  real ( kind = 8 ) z

  den = sqrt ( a * a + b * b )
  c = a / den
  s = b / den

  do j = i, n
    y = r(i,j)
    z = r(i+1,j)
    r(i,j) = c * y - s * z
    r(i+1,j) = s * y + c * z
  end do

  return
end subroutine qraux2
subroutine qrupdt ( nr, n, a, u, v )

!*****************************************************************************80
!
!! QRUPDT updates a QR factorization.
!
!  Discussion:
!
!    The routine finds an orthogonal N by N matrix Q* and an upper triangular
!    N by N matrix R* such that (Q*)(R*) = R + U*V'
!
!  Parameters:
!
!    Input, integer NR, the row dimension of the matrix.
!
!    Input, integer N, the order of the matrix.
!
!    Input/output, real ( kind = 8 ) A(NR,N), on input, contains the original QR
!    factorization.  On output, contains the revised factorization.
!
!    Input, real ( kind = 8 ) U(N), V(N), vectors that describe the rank
!    one update applied to the original matrix A.
!
  implicit none

  integer n
  integer nr

  real ( kind = 8 ) a(nr,n)
  integer i
  integer k
  real ( kind = 8 ) t1
  real ( kind = 8 ) t2
  real ( kind = 8 ) u(n)
  real ( kind = 8 ) v(n)
!
!  Determine the last non-zero in U.
!
  k = n

  do while ( u(k) == 0.0D+00 .and. 1 < k )
    k = k - 1
  end do
!
!  (k-1) Jacobi rotations transform
!    r + u(v+) --> (r*) + ( u(1) * e1 ) (v+)
!  which is upper Hessenberg
!
  if ( 1 < k ) then

    do i = k-1, 1, -1

      if ( u(i) == 0.0D+00 ) then
        call qraux1 ( nr, n, a, i )
        u(i) = u(i+1)
      else
        call qraux2 ( nr, n, a, i, u(i), -u(i+1) )
        u(i) = sqrt ( u(i) * u(i) + u(i+1) * u(i+1) )
      end if

    end do

  end if
!
!  R <-- R + ( u(1) * e1 ) (v+)
!
  a(1,1:n) = a(1,1:n) + u(1) * v(1:n)
!
!  (k-1) Jacobi rotations transform upper Hessenberg R
!  to upper triangular (R*)
!
    do i = 1, k-1

      if ( a(i,i) == 0.0D+00 ) then
        call qraux1 ( nr, n, a, i )
      else
        t1 = a(i,i)
        t2 = -a(i+1,i)
        call qraux2 ( nr, n, a, i, t1, t2 )
      end if

    end do

  return
end subroutine qrupdt
subroutine r8_swap ( x, y )

!*****************************************************************************80
!
!! R8_SWAP swaps two R8's.
!
!  Modified:
!
!    22 December 2000
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input/output, real ( kind = 8 ) X, Y.  On output, the values of X and
!    Y have been interchanged.
!
  implicit none

  real ( kind = 8 ) x
  real ( kind = 8 ) y
  real ( kind = 8 ) z

  z = x
  x = y
  y = z

  return
end subroutine r8_swap
subroutine result ( nr, n, x, f, g, a, p, itncnt, iflg, ipr )

!*****************************************************************************80
!
!! RESULT prints information about the optimization process.
!
!  Parameters:
!
!    Input, integer NR, the row dimension of the matrix.
!
!    Input, integer N, the dimension of the problem.
!
!    Input, real ( kind = 8 ) X(N), the current iterate.
!
!    Input, real ( kind = 8 ) F, the function value at X.
!
!    Input, real ( kind = 8 ) G(N), the gradient at X.
!
!    Input, real ( kind = 8 ) A(NR,N), the N by N Hessian matrix at X.
!
!    Input, real ( kind = 8 ) P(N), the step taken.
!
!    Input, integer ITNCNT, the iteration number.
!
!    Input, integer IFLG, the flag controlling the amount of printout.
!
!    Input, integer IPR, the device to which to send output.
!
  implicit none

  integer n
  integer nr

  real ( kind = 8 ) a(nr,n)
  real ( kind = 8 ) f
  real ( kind = 8 ) g(n)
  integer i
  integer iflg
  integer ipr
  integer itncnt
  real ( kind = 8 ) p(n)
  real ( kind = 8 ) x(n)

  write ( ipr, 903 ) itncnt

  if ( iflg /= 0 ) then
    write ( ipr, * ) ' result       step'
    write ( ipr,905) p(1:n)
  end if

  write ( ipr, * ) ' result       x(k)'
  write ( ipr, 905) x(1:n)
  write ( ipr, * ) ' result     function at x(k)'
  write ( ipr, 905) f
  write ( ipr, * ) ' result       gradient at x(k)'
  write ( ipr, 905) g(1:n)

  if ( iflg /= 0 ) then

    write ( ipr, * ) ' result       Hessian at x(k)'
    do i = 1, n
      write ( ipr, 900) i
      write ( ipr, 902) a(i,1:i)
    end do

  end if

  return

  900 format(' result     row',i5)
  902 format(' result       ',5(2x,e20.13))
  903 format(/'0result    iterate k=',i5)
  905 format(' result               ',5(2x,e20.13) )
end subroutine result
subroutine secfac ( nr, n, x, g, a, xpls, gpls, epsm, itncnt, rnf, &
  iagflg, noupdt, s, y, u, w )

!*****************************************************************************80
!
!! SECFAC updates the hessian by the BFGS factored method.
!
!  Parameters:
!
!    Input, integer NR, the row dimension of the matrix.
!
!    Input, integer N, the dimension of the problem.
!
!    Input, real ( kind = 8 ) X(N), the old iterate, X[K-1].
!
!    Input, real ( kind = 8 ) G(N), the gradient or an approximation,
!    at the old iterate.
!
!    Input/output, real ( kind = 8 ) A(NR,N).
!    On input, the Cholesky decomposition of hessian in lower part and diagonal.
!    On output, the updated Cholesky decomposition of hessian
!    in lower triangular part and diagonal
!
!    Input, real ( kind = 8 ) XPLS(N), the new iterate, X[K].
!
!    Input, real ( kind = 8 ) GPLSN(N), gradient, or an approximation,
!    at the new iterate.
!
!    Input, real ( kind = 8 ) EPSM, the machine epsilon.
!
!    Input, integer ITNCNT, the iteration count.
!
!    Input, real ( kind = 8 ) RNF, the relative noise in the optimization
!    function FCN.
!
!    Input, integer IAGFLG, 1 if analytic gradient supplied.
!
!    Input/output, logical NOUPDT, is TRUE if there has been no update
!    yet.  The user should retain the output value between successive
!    calls.
!
!    Workspace, real ( kind = 8 ) S(N).
!
!    Workspace, real ( kind = 8 ) Y(N).
!
!    Workspace, real ( kind = 8 ) U(N).
!
!    Workspace, real ( kind = 8 ) W(N).
!
  implicit none

  integer n
  integer nr

  real ( kind = 8 ) a(nr,n)
  real ( kind = 8 ) alp
  real ( kind = 8 ) den1
  real ( kind = 8 ) den2
  real ( kind = 8 ) dnrm2
  real ( kind = 8 ) epsm
  real ( kind = 8 ) g(n)
  real ( kind = 8 ) gpls(n)
  integer i
  integer iagflg
  integer itncnt
  integer j
  logical noupdt
  real ( kind = 8 ) reltol
  real ( kind = 8 ) rnf
  real ( kind = 8 ) s(n)
  logical skpupd
  real ( kind = 8 ) snorm2
  real ( kind = 8 ) u(n)
  real ( kind = 8 ) w(n)
  real ( kind = 8 ) x(n)
  real ( kind = 8 ) xpls(n)
  real ( kind = 8 ) y(n)
  real ( kind = 8 ) ynrm2

  if ( itncnt == 1 ) then
    noupdt = .true.
  end if

  s(1:n) = xpls(1:n) - x(1:n)
  y(1:n) = gpls(1:n) - g(1:n)

  den1 = dot_product ( s, y )

  snorm2 = dnrm2 ( n, s, 1)

  ynrm2 = dnrm2 ( n, y, 1)

  if ( den1 < sqrt ( epsm ) * snorm2 * ynrm2 ) then
    return
  end if

  call mvmltu ( nr, n, a, s, u )

  den2 = dot_product ( u, u )
!
!  L <-- sqrt ( den1 / den2 ) * L
!
  alp = sqrt ( den1 / den2 )

  if ( noupdt ) then

    u(1:n) = alp * u(1:n)

    do j = 1, n
      do i = j, n
        a(i,j) = alp * a(i,j)
      end do
    end do

    noupdt = .false.
    den2 = den1
    alp = 1.0D+00

  end if

  skpupd = .true.
!
!  W = l(l+)s = hs
!
  call mvmltl ( nr, n, a, u, w )
  i = 1

  if ( iagflg == 0 ) then
    reltol = sqrt ( rnf )
  else
    reltol = rnf
  end if

60  continue

  if ( i <= n .and. skpupd ) then

    if ( abs ( y(i) - w(i) ) < reltol * &
      max ( abs ( g(i) ), abs ( gpls(i) ) ) ) then
      i = i + 1
    else
      skpupd = .false.
    end if
    go to 60
  end if

  if ( skpupd ) then
    return
  end if
!
!  W = y-alp*l(l+)s
!
  w(1:n) = y(1:n) - alp * w(1:n)
!
!  ALP = 1 / sqrt ( den1 * den2 )
!
  alp = alp / den1
!
!  U = (l+) / sqrt ( den1 * den2 ) = (l+)s/ sqrt ( ( y+ ) s * (s+) l (l+) s )
!
  u(1:n) = alp * u(1:n)
!
!  Copy L into upper triangular part.  Zero L.
!
  do i = 2, n
    do j = 1, i-1
      a(j,i) = a(i,j)
      a(i,j) = 0.0D+00
    end do
  end do
!
!  Find Q, (l+) such that  q(l+) = (l+) + u(w+)
!
  call qrupdt ( nr, n, a, u, w )
!
!  Upper triangular part and diagonal of a now contain updated
!  Cholesky decomposition of hessian.  Copy back to lower triangular part.
!
  do i = 2, n
    do j = 1, i-1
      a(i,j) = a(j,i)
    end do
  end do

  return
end subroutine secfac
subroutine secunf ( nr, n, x, g, a, udiag, xpls, gpls, epsm, itncnt, &
  rnf, iagflg, noupdt, s, y, t )

!*****************************************************************************80
!
!! SECUNF updates a Hessian matrix by the BFGS unfactored method.
!
!  Parameters:
!
!    Input, integer NR, the row dimension of the matrix.
!
!    Input, integer N, the dimension of the problem.
!
!    Input, real ( kind = 8 ) X(N), the old iterate, X[K-1].
!
!    Input, real ( kind = 8 ) G(N), the gradient, or an approximate value,
!    at the  old iterate.
!
!    Input/output, real ( kind = 8 ) A(NR,N).
!    on entry: approximate hessian at old iterate
!    in upper triangular part (and udiag)
!    on exit:  updated approx hessian at new iterate
!    in lower triangular part and diagonal
!    [lower triangular part of symmetric matrix]
!
!    Input, real ( kind = 8 ) UDIAG(N), the diagonal entries of the hessian.
!
!    Input, real ( kind = 8 ) XPLS(N), the new iterate, X[K].
!
!    Input, real ( kind = 8 ) GPLS(N), the gradient or an approximate value, at
!    the new iterate
!
!    Input, real ( kind = 8 ) EPSM, the machine epsilon.
!
!    Input, integer ITNCNT, the iteration count.
!
!    Input, real ( kind = 8 ) RNF, the relative noise in the optimization
!    function.
!
!    Input, integer IAGFLG, =1 if analytic gradient supplied, =0 otherwise
!
!    Input/output, logical NOUPDT, TRUE if no update yet.
!    [retain value between successive calls]
!
!    Workspace, real ( kind = 8 ) S(N).
!
!    Workspace, real ( kind = 8 ) Y(N).
!
!    Workspace, real ( kind = 8 ) T(N).
!
  implicit none

  integer n
  integer nr

  real ( kind = 8 ) a(nr,n)
  real ( kind = 8 ) den1
  real ( kind = 8 ) den2
  real ( kind = 8 ) dnrm2
  real ( kind = 8 ) epsm
  real ( kind = 8 ) g(n)
  real ( kind = 8 ) gam
  real ( kind = 8 ) gpls(n)
  integer i
  integer iagflg
  integer itncnt
  integer j
  logical noupdt
  real ( kind = 8 ) rnf
  real ( kind = 8 ) s(n)
  logical skpupd
  real ( kind = 8 ) snorm2
  real ( kind = 8 ) t(n)
  real ( kind = 8 ) tol
  real ( kind = 8 ) udiag(n)
  real ( kind = 8 ) x(n)
  real ( kind = 8 ) xpls(n)
  real ( kind = 8 ) y(n)
  real ( kind = 8 ) ynrm2
!
!  Copy hessian in upper triangular part and UDIAG to
!  lower triangular part and diagonal.
!
  do j = 1, n
    a(j,j) = udiag(j)
    do i = j+1, n
      a(i,j) = a(j,i)
    end do
  end do

  if ( itncnt == 1 ) then
    noupdt=.true.
  end if

  s(1:n) = xpls(1:n) - x(1:n)
  y(1:n) = gpls(1:n) - g(1:n)

  den1 = dot_product ( s, y )

  snorm2 = dnrm2 ( n, s, 1 )

  ynrm2 = dnrm2 ( n, y, 1 )

  if ( den1 < sqrt ( epsm ) * snorm2 * ynrm2 ) then
    return
  end if

  call mvmlts ( nr, n, a, s, t )

  den2 = dot_product ( s, t )

  if ( noupdt ) then
!
!  H <-- [(s+)y/(s+)hs]h
!
    gam = den1 / den2
    den2 = gam * den2
    do j = 1, n
      t(j) = gam * t(j)
      do i = j, n
        a(i,j) = gam * a(i,j)
      end do
    end do
    noupdt = .false.

  end if

  skpupd = .true.
!
!  Check update condition on row I.
!
  do i = 1, n

    tol = rnf * max ( abs ( g(i) ), abs ( gpls(i) ) )
    if ( iagflg == 0 ) then
      tol = tol / sqrt ( rnf )
    end if

    if ( tol <= abs ( y(i) - t(i) ) ) then
      skpupd = .false.
      exit
    end if

  end do

  if ( skpupd ) then
    return
  end if
!
!  BFGS update
!
  do j = 1, n
    do i = j, n
      a(i,j) = a(i,j) + y(i) * y(j) / den1 - t(i) * t(j) / den2
    end do
  end do

  return
end subroutine secunf
subroutine sndofd ( nr, n, xpls, fcn, fpls, a, sx, rnoise, stepsz, anbr )

!*****************************************************************************80
!
!! SNDOFD approximates a Hessian with a second order finite difference.
!
!  Parameters:
!
!    Input, integer NR, the row dimension of the matrix.
!
!    Input, integer N, the dimension of the problem.
!
!    Input, real ( kind = 8 ) XPLS(N), the new iterate, X[K].
!
!    Input, external FCN, the name of the subroutine to evaluate the function,
!    of the form
!
!      subroutine fcn ( n, x, f )
!      integer n
!      real x(n)
!      real f
!
!    Input, real ( kind = 8 ) FPLS, the function value at the new iterate.
!
!    Output, real ( kind = 8 ) A(NR,N), the N by N  finite difference
!    approximation to the hessian matrix.  Only the lower triangular matrix and
!    diagonal are returned.
!
!    Input, real ( kind = 8 ) SX(N), the diagonal scaling for X.
!
!    Input, real ( kind = 8 ) RNOISE, the relative noise in the function.
!
!    Workspace, real ( kind = 8 ) STEPSZ(N), used for the stepsize.
!
!    Workspace, real ( kind = 8 ) ANBR(N), holds neighbors.
!
  implicit none

  integer n
  integer nr

  real ( kind = 8 ) a(nr,n)
  real ( kind = 8 ) anbr(n)
  external fcn
  real ( kind = 8 ) fhat
  real ( kind = 8 ) fpls
  integer i
  integer j
  real ( kind = 8 ) ov3
  real ( kind = 8 ) rnoise
  real ( kind = 8 ) stepsz(n)
  real ( kind = 8 ) sx(n)
  real ( kind = 8 ) xpls(n)
  real ( kind = 8 ) xtmpi
  real ( kind = 8 ) xtmpj
!
!  Find I-th stepsize and evaluate neighbor in direction of I-th unit vector.
!
  ov3 = 1.0D+00 / 3.0D+00

  do i = 1, n
    stepsz(i) = rnoise**ov3 * max ( abs ( xpls(i) ), 1.0D+00 / sx(i) )
    xtmpi = xpls(i)
    xpls(i) = xtmpi + stepsz(i)
    call fcn ( n, xpls, anbr(i) )
    xpls(i) = xtmpi
  end do
!
!  Calculate column I of A.
!
  do i = 1, n

    xtmpi = xpls(i)
    xpls(i) = xtmpi + 2.0D+00 * stepsz(i)
    call fcn ( n, xpls, fhat )
    a(i,i) = ( ( fpls - anbr(i) ) &
      + ( fhat - anbr(i) ) ) / ( stepsz(i) * stepsz(i) )
!
!  Calculate sub-diagonal elements of column.
!
    if ( i /= n ) then

      xpls(i) = xtmpi + stepsz(i)

      do j = i + 1, n
        xtmpj = xpls(j)
        xpls(j) = xtmpj + stepsz(j)
        call fcn ( n, xpls, fhat )
        a(j,i) = ( ( fpls - anbr(i) ) + ( fhat - anbr(j) ) ) &
          / ( stepsz(i) * stepsz(j) )
        xpls(j) = xtmpj
      end do

    end if

    xpls(i) = xtmpi

  end do

  return
end subroutine sndofd
subroutine tregup ( nr, n, x, f, g, a, fcn, sc, sx, nwtake, stepmx, steptl, &
  dlt, iretcd, xplsp, fplsp, xpls, fpls, mxtake, ipr, method, udiag )

!*****************************************************************************80
!
!! TREGUP decides whether to accept the next optimization iterate.
!
!  Parameters:
!
!    Input, integer NR, the row dimension of the matrix.
!
!    Input, integer N, the dimension of the problem.
!
!    Input, real ( kind = 8 ) X(N), the old iterate X[K-1].
!
!    Input, real ( kind = 8 ) F, the function value at the old iterate.
!
!    Input, real ( kind = 8 ) G(N), the gradient at the old iterate, or
!    an approximate value of it.
!
!    Input, real ( kind = 8 ) A(NR,N), the Cholesky decomposition of hessian in
!    lower triangular part and diagonal.  Hessian or approximation in
!    upper triangular part.
!
!    Input, external FCN, the name of the subroutine to evaluate the function,
!    of the form
!
!      subroutine fcn ( n, x, f )
!      integer n
!      real ( kind = 8 ) x(n)
!      real ( kind = 8 ) f
!
!    Input, real ( kind = 8 ) SC(N), the current step.
!
!    Input, real ( kind = 8 ) SX(N), the diagonal scaling matrix for X.
!
!    Input, logical NWTAKE, is TRUE if a Newton step is taken.
!
!    Input, real ( kind = 8 ) STEPMX, the maximum allowable step size.
!
!    Input, real ( kind = 8 ) STEPTL, the relative step size at which successive
!    iterates are considered close enough to terminate algorithm.
!
!    Input/output, real ( kind = 8 ) DLT, the trust region radius.
!
!    Input/output, integer IRETCD, the status code.
!    0, xpls accepted as next iterate;  dlt trust region for next iteration.
!    1, xpls unsatisfactory but accepted as next iterate because
!      xpls-x < smallest allowable step length.
!    2, f(xpls) too large.  continue current iteration with new reduced dlt.
!    3, f(xpls) sufficiently small, but quadratic model predicts f(xpls)
!      sufficiently well to continue current iteration with new doubled dlt.
!
!    Workspace, real ( kind = 8 ) XPLSP(N), [value needs to be retained between
!    succesive calls of k-th global step].
!
!    Worskpace, real ( kind = 8 ) FPLSP, [retain value between successive
!    calls].
!
!    Output, real ( kind = 8 ) XPLS(N), the new iterate x[k].
!
!    Output, real ( kind = 8 ) FPLS, the function value at the new iterate.
!
!    Output, logical MXTAKE, is TRUE if a step of maximum length was taken.
!
!    Input, integer IPR, the device to which to send output.
!
!    Input, integer METHOD, the algorithm to use.
!    1, line search,
!    2, double dogleg,
!    3, More-Hebdon.
!
!    Input, real ( kind = 8 ) UDIAG(N), the diagonal of hessian in a(.,.)
!
  implicit none

  integer n
  integer nr

  real ( kind = 8 ) a(nr,n)
  real ( kind = 8 ) dlt
  real ( kind = 8 ) dltf
  real ( kind = 8 ) dltfp
  real ( kind = 8 ) dltmp
  real ( kind = 8 ) f
  external fcn
  real ( kind = 8 ) fpls
  real ( kind = 8 ) fplsp
  real ( kind = 8 ) g(n)
  integer i
  integer ipr
  integer iretcd
  integer j
  integer method
  logical mxtake
  logical nwtake
  real ( kind = 8 ) rln
  real ( kind = 8 ) sc(n)
  real ( kind = 8 ) slp
  real ( kind = 8 ) stepmx
  real ( kind = 8 ) steptl
  real ( kind = 8 ) sx(n)
  real ( kind = 8 ) temp
  real ( kind = 8 ) udiag(n)
  real ( kind = 8 ) x(n)
  real ( kind = 8 ) xpls(n)
  real ( kind = 8 ) xplsp(n)

  mxtake = .false.
  xpls(1:n) = x(1:n) + sc(1:n)
  call fcn ( n, xpls, fpls )
  dltf = fpls - f
  slp = dot_product ( g(1:n), sc(1:n) )
!
!  Next statement added for case of compilers which do not optimize
!  evaluation of next "if" statement (in which case fplsp could be
!  undefined).
!
  if ( iretcd == 4 ) then
    fplsp = 0.0D+00
  end if
!
!  Reset XPLS to XPLSP and terminate global step.
!
  if ( iretcd == 3 .and. ( fplsp <= fpls .or. 1.0D-04 * slp < dltf ) ) then
    iretcd = 0
    xpls(1:n) = xplsp(1:n)
    fpls = fplsp
    dlt = 0.5D+00 * dlt
    return
  end if

  if ( dltf <= 1.0D-04 * slp ) then
    go to 170
  end if

  rln = 0.0D+00

  do i = 1, n

    rln = max (                                        &
                rln,                                   &
                abs ( sc(i) ) / max (                  &
                                      abs ( xpls(i) ), &
                                      1.0D+00 / sx(i)  &
                                    )                  &
              )
  end do
!
!  Cannot find satisfactory xpls sufficiently distinct from x
!
  if ( rln < steptl ) then
    iretcd = 1
    return
  end if
!
!  Reduce trust region and continue global step
!
        iretcd = 2
        dltmp = -slp * dlt / ( 2.0D+00 * ( dltf - slp ) )

        if ( 0.1D+00 * dlt <= dltmp ) then
          go to 155
        end if

          dlt = 0.1D+00 * dlt
          go to 160

  155   continue
        dlt = dltmp

  160   continue
        return
!
!  FPLS sufficiently small
!
  170     continue

      dltfp = 0.0D+00

      if ( method == 2 ) then

        do i = 1, n
          temp = dot_product ( sc(i:n), a(i:n,i) )
          dltfp = dltfp + temp**2
        end do

      else

        do i = 1, n
          dltfp = dltfp + udiag(i) * sc(i) * sc(i)

          temp = 0.0D+00
          do j = i+1, n
            temp = temp + a(i,j) * sc(i) * sc(j)
          end do
          dltfp = dltfp + 2.0D+00 * temp
        end do

      end if

      dltfp = slp + dltfp / 2.0D+00

      if ( iretcd == 2 .or. &
         0.1D+00 * abs ( dltf ) < abs ( dltfp - dltf ) .or. &
         nwtake .or. &
         0.99D+00 * stepmx < dlt ) then
        go to 210
      end if
!
!  Double trust region and continue global step
!
        iretcd = 3
        xplsp(1:n) = xpls(1:n)
        fplsp = fpls
        dlt = min ( 2.0D+00 * dlt, stepmx )
        return
!
!  Accept XPLS as the next iterate.  Choose the new trust region.
!
  210       continue

  iretcd = 0

  if ( 0.99D+00 * stepmx < dlt ) then
    mxtake = .true.
  end if

  if ( dltf < 0.1D+00 * dltfp ) then
    if ( dltf <= 0.75D+00 * dltfp ) then
      dlt = min ( 2.0D+00 * dlt, stepmx )
    end if
  else
    dlt = 0.5D+00 * dlt
  end if

  return
end subroutine tregup
end module uncmin_int
subroutine d1fcn ( n, x, g )

!*****************************************************************************80
!
!! D1FCN is a dummy routine for evaluating the gradient vector.
!
!  Discussion:
!
!    We assume that F is a scalar function of N variables X.  The routine
!    is to compute the vector G where G(I) = d F/d X(I).
!
!  Modified:
!
!    16 April 2001
!
!  Parameters:
!
!    Input, integer N, the dimension of X, and order of A.
!
!    Input, real ( kind = 8 ) X(N), the point at which the gradient
!    is to be evaluated.
!
!    Output, real ( kind = 8 ) G(N), the gradient vector..
!
  implicit none

  integer n

  real ( kind = 8 ) g(n)
  real ( kind = 8 ) x(n)

  write ( *, '(a)' ) ' '
  write ( *, '(a)' ) 'D1FCN - Fatal error!'
  write ( *, '(a)' ) '  This is a dummy routine.'
  write ( *, '(a)' ) '  The user is required to replace it with a'
  write ( *, '(a)' ) '  routine that computes the gradient of F.'

  stop
end subroutine d1fcn
subroutine d2fcn ( nr, n, x, a )

!*****************************************************************************80
!
!! D2FCN is a dummy version of a routine that computes the second derivative.
!
!  Discussion:
!
!    We assume that F is a scalar function of N variables X.  The routine
!    is to compute the matrix H where H(I,J) = d d F / d X(I) d X(J).
!
!  Modified:
!
!    16 April 2001
!
!  Parameters:
!
!    Input, integer NR, the leading dimension of A, which must be at
!    least N.
!
!    Input, integer N, the dimension of X, and order of A.
!
!    Input, real ( kind = 8 ) X(N), the point at which the Hessian matrix
!    is to be evaluated.
!
!    Output, real ( kind = 8 ) A(NR,N), the N by N Hessian matrix.
!
  implicit none

  integer n
  integer nr

  real ( kind = 8 ) a(nr,n)
  real ( kind = 8 ) x(n)

  write ( *, '(a)' ) ' '
  write ( *, '(a)' ) 'D2FCN - Fatal error!'
  write ( *, '(a)' ) '  This is a dummy routine.'
  write ( *, '(a)' ) '  The user is required to replace it with a'
  write ( *, '(a)' ) '  routine that computes the Hessian matrix of F.'

  stop
end subroutine d2fcn
function dnrm2 ( n, x, incx )

!*****************************************************************************80
!
!! DNRM2 returns the euclidean norm of a vector.
!
!  Discussion:
!
!     DNRM2 ( X ) = sqrt ( X' * X )
!
!  Author:
!
!    Sven Hammarling
!
!  Reference:
!
!    Jack Dongarra, Cleve Moler, Jim Bunch, Pete Stewart,
!    LINPACK User's Guide,
!    SIAM, 1979.
!
!    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
!    Basic Linear Algebra Subprograms for Fortran Usage,
!    Algorithm 539,
!    ACM Transactions on Mathematical Software,
!    Volume 5, Number 3, September 1979, pages 308-323.
!
!  Parameters:
!
!    Input, integer N, the number of entries in the vector.
!
!    Input, real ( kind = 8 ) X(*), the vector whose norm is to be computed.
!
!    Input, integer INCX, the increment between successive entries of X.
!
!    Output, real ( kind = 8 ) DNRM2, the Euclidean norm of X.
!
  implicit none

  real ( kind = 8 ) absxi
  real ( kind = 8 ) dnrm2
  integer incx
  integer ix
  integer n
  real ( kind = 8 ) norm
  real ( kind = 8 ) scale
  real ( kind = 8 ) ssq
  real ( kind = 8 ) x(*)

  if ( n < 1 .or. incx < 1 ) then

    norm  = 0.0D+00

  else if ( n == 1 ) then

    norm  = abs ( x(1) )

  else

    scale = 0.0D+00
    ssq = 1.0D+00

    do ix = 1, 1 + ( n - 1 )*incx, incx
      if ( x(ix) /= 0.0D+00 ) then
        absxi = abs ( x(ix) )
        if ( scale < absxi ) then
          ssq = 1.0D+00 + ssq * ( scale / absxi )**2
          scale = absxi
        else
          ssq = ssq + ( absxi / scale )**2
        end if
      end if
    end do
    norm  = scale * sqrt( ssq )
  end if

  dnrm2 = norm

  return
end function dnrm2
