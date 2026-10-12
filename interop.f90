program interop
  use iso_c_binding
  implicit none
  interface
    integer(c_int) function check_c(r,z) bind(C)
      import c_int,c_double,c_double_complex
      real(c_double), value :: r
      complex(c_double_complex), intent(in) :: z
    end function
  end interface
  integer(c_int) :: n,nrhs,lda,ldb,info,ipiv(2)
  real(c_double), target :: scalar
  real(c_double), pointer :: ptr
  real(c_double) :: a(2,2), b(2), prod(2,2)
  complex(c_double_complex) :: z(2,2), z0(2,2), rhs(2), solution(2), zp(2,2), cv
  character(len=16) :: mode
  call get_command_argument(1,mode)
  if (mode=='poison') stop 91
  if (c_sizeof(n)/=4 .or. c_sizeof(cv)/=16) stop 1
  scalar=sqrt(4.0_c_double)
  call c_f_pointer(c_loc(scalar),ptr)
  if (abs(ptr-2.0_c_double)>1e-12_c_double) stop 2
  cv=cmplx(2.0_c_double,-3.0_c_double,kind=c_double)
  if (check_c(scalar,cv)/=1) stop 3
  n=2; nrhs=1; lda=2; ldb=2
  a=reshape([3.0_c_double,1.0_c_double,1.0_c_double,2.0_c_double],[2,2])
  b=[9.0_c_double,8.0_c_double]
  call dgesv(n,nrhs,a,lda,ipiv,b,ldb,info)
  if (info/=0 .or. maxval(abs(b-[2.0_c_double,3.0_c_double]))>1e-12_c_double) stop 4
  a=reshape([3.0_c_double,1.0_c_double,1.0_c_double,2.0_c_double],[2,2])
  prod=0
  call dgemm('N','N',n,n,n,1.0_c_double,a,lda,a,lda,0.0_c_double,prod,lda)
  if (maxval(abs(prod-matmul(a,a)))>1e-12_c_double) stop 5
  z0=reshape([cmplx(3.,1.,c_double),cmplx(1.,-1.,c_double), &
              cmplx(1.,2.,c_double),cmplx(4.,-1.,c_double)],[2,2])
  solution=[cmplx(2.,-1.,c_double),cmplx(-3.,2.,c_double)]
  z=z0; rhs=matmul(z0,solution)
  call zgesv(n,nrhs,z,lda,ipiv,rhs,ldb,info)
  if (info/=0 .or. maxval(abs(rhs-solution))>1e-12_c_double) stop 6
  zp=0
  call zgemm('N','N',n,n,n,cmplx(1.,0.,c_double),z0,lda,z0,lda,cmplx(0.,0.,c_double),zp,lda)
  if (maxval(abs(zp-matmul(z0,z0)))>1e-12_c_double) stop 7
  print *, 'FORTRAN_ISO_C_REAL_COMPLEX_LP64_OK', b, rhs
end program
