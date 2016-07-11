% Initial and Final Encodings

# Introduction

Code is Data, and Data is Code

Changing how you represent your data or your program can yield
practical improvements.

# The Basic Idea

~~~~ {.haskell}
data List a = Nil
            | Cons a (List a)

-- data [a] = [] | a : [a]
~~~~

 - Start at the head
 - Do something on each element in order

# The Basic Idea

~~~~ {.haskell}
data List a = Nil
            | Cons a (List a)

-- data [a] = [] | a : [a]
~~~~

~~~~ {.haskell}
(++) :: [a] -> [a] -> [a]
[]     ++ bs = bs
(a:as) ++ bs = a : (as ++ bs)

-- call `:` once per element of the first list, every time you call `++`!
~~~~

# The Basic Idea

What if we turned the list inside-out?

~~~~ {.haskell}
newtype DList = DList { runDList :: [a] -> [a] }
~~~~

~~~~ {.haskell}
toDList :: [a] -> DList a
toDList lst = DList (lst ++)
~~~~

~~~~ {.haskell}
fromDList :: DList a -> [a]
fromDList (DList f) = f []
~~~~

~~~~ {.haskell}
cons :: a -> DList a -> DList a
cons x (DList xs) = DList $ (x :) . xs

-- cons x (DList xs) = DList $ \g -> x : (xs g)
~~~~

# The Basic Idea

~~~~ {.haskell}
appendDList :: DList a -> DList a -> DList a
appendDList (DList xs) (DList ys) = DList $ xs . ys
-- appendDList (DList xs) (DList ys) = DList $ \g -> xs (ys g)

-- call (.) once per call to `appendDList`!
~~~~

~~~~ {.haskell}
(++) :: [a] -> [a] -> [a]
[]     ++ bs = bs
(a:as) ++ bs = a : (as ++ bs)
~~~~

~~~~ {.haskell}
-- convert a series of appends back to an ordinary list

(as, bs, cs) = (toDList a, toDList b, toDList c)

as `appendDList` bs `appendDList` cs      -- expand appendDList (associativity doesn't matter)
DList $ as . bs . cs                      -- rewrite each DList in terms of `toDList`
DList $ (a ++) . (b ++) . (c ++)          -- run `fromDList`
(a ++) . (b ++) . (c ++) $ []             -- apply first function
(a ++) . (b ++) $ c ++ []                 -- traverse `c` once when evaluating `++`
(a ++) . (b ++) $ c                       -- apply second function
(a ++) $ (b ++ c)                         -- traverse `b` once when evaluating `++`
(a ++) $ bc                               -- apply third function
a ++ bc                                   -- traverse `a` once when evaluating `++`
abc                                       -- concrete list in O(length abc)
~~~~

# A Less Basic Idea

~~~~ {.haskell}
{-# LANGUAGE RankNTypes #-}  -- so we can use 'forall'

newtype Church a = Church {
  runChurch :: forall r. (a -> r -> r)    -> r   -> r
               --          cons              nil    result
  }

convert :: [a] -> Church a
convert lst = Church $ \c n -> foldr c n lst

excommunicate :: Church a -> [a]
excommunicate (Church ch) = ch (:) []

cons :: a -> Church a -> Church a
cons x (Church ch) = Church $ \c -> c x . ch c
-- cons x (Church ch) = Church $ \c n -> c x (ch c n)

appendChurch :: Church a -> Church a -> Church a
appendChurch (Church fx) (Church fy) = Church $ \c -> fx c . fy c
-- appendChurch xs ys = Church $ \c n -> runChurch xs c $ runChurch ys c n

-- run one function composition per call to `appendChurch'`!
~~~~

~~~~ {.haskell}
foldr :: Foldable t => (a -> b -> b) -> b -> t a -> b
                   c :: a -> r -> r           -- the folding function
                                   n :: r     -- the terminating value
~~~~

# A Less Basic Idea

More general than `DList`!

~~~~ {.haskell}
data Tree a = Node (Tree a) a (Tree a)
            | Tip

newtype Cross a = Cross {
    runCross :: forall r. (r -> a -> r -> r) -> r -> r
    }

tip :: Cross a
tip = Cross $ \_ t -> t

node :: Cross a -> a -> Cross a -> Cross a
node (Cross l) x (Cross r) = Cross $ \n t -> n (l n t) x (r n t)

convertTree :: Tree a -> Cross a
convertTree Tip = tip
convertTree (Node l x r) = node (convertTree l) x (convertTree r)

excommunicateTree :: Cross a -> Tree a
excommunicateTree (Cross x) = x Node Tip
~~~~

Only the conversion need know about the ADT!

# A Less Basic Idea

Applies to very pragmatic problems, e.g. you don't have proper ADTs.
From `#bfpg` on freenode:

> 04:38:09          bkolera | It wouldn't be a big deal if we didn't encode sum types as sub types in scala, but ... >_>
>
> 04:42:56          georgew | You could just Church encode?

# (Embedded) Domain-Specific Languages

 * Separate the problem domain from the software engineering

 * More accessible to domain experts than the general-purpose host
   language (or "metalanguage")

 * Reuse the metalanguage's parser

# The Multiple Interpretations Problem

Ways to interpret a program

  * Run the program and get its result

# The Multiple Interpretations Problem

Ways to interpret a program

  * Run the program and get its result

  * Pretty-print or serialize the program

  * Perform optimizations before executing the program

  * Run additional checks or pre-run tests

  * Generate code

  * Execute using alternative evaluation strategies

# The Expression Problem

> The Expression Problem is a new name for an old problem.  The goal
> is to define a datatype by cases, where one can add new cases to the
> datatype and new functions over the datatype, without recompiling
> existing code, and while retaining static type safety (e.g., no
> casts).

  -- Philip Wadler, Nov. 1998

# The Static Safety Problem

What are the hard parts of implementing a language?

 - parsers

 - code generation/evaluation

 - type checkers

Our interpreter should let us reuse all three!

# Solving All The Problems At Once

Simply-typed lambda calculus: small but higher-order.

 * ADTs

 * tagless with GADTs

 * final encoding

 * the "Finally Tagless" approach

# A Simple (But Powerful) DSL

STLC, with de Bruijn indices

~~~~ {.haskell}
-- Simply-typed lambda calculus terms
data STLCTerm = SVar Var
              | SNum Int
              | SApp STLCTerm STLCTerm
              | SLam STLCTerm
              deriving Show

-- Representation of evaluation environments
data Var = VZero | VSucc Var deriving Show

-- We need a way to look up variables in our environment.
lkup :: Var -> [x] -> x
lkup VZero (x:_) = x
lkup (VSucc v) (_:moar) = lkup v moar
lkup _ _ = error "sorry, dawg; I can't interpret an open term!"
~~~~

~~~~ {.haskell}
evalSTLC env (SVar v) = lkup v env
evalSTLC env (SNum i) = i
evalSTLC env (SApp f x) = (evalSTLC env f) (evalSTLC env x)
evalSTLC env (SLam b) = \x -> evalSTLC (x : env) b
~~~~

~~~~ {.haskell}
testSTLC = SApp (SLam (SVar VZero)) (SNum 22)
~~~~

# A Simple (But Powerful) DSL

What a puzzling error!

~~~~ {.text}
[1 of 1] Compiling FinallyTagless   ( FinallyTagless.hs, interpreted )

FinallyTagless.hs:122:27:
    Couldn't match expected type ‘Int -> Int’ with actual type ‘Int’
    The function ‘evalSTLC’ is applied to three arguments,
    but its type ‘[Int] -> STLCTerm -> Int’ has only two
    In the expression: (evalSTLC env f) (evalSTLC env x)
    In an equation for ‘evalSTLC’:
        evalSTLC env (SApp f x) = (evalSTLC env f) (evalSTLC env x)

FinallyTagless.hs:123:25:
    Couldn't match expected type ‘Int -> Int’ with actual type ‘Int’
    The lambda expression ‘\ x -> evalSTLC (x : env) b’
    has one argument,
    but its type ‘Int’ has none
    In the expression: \ x -> evalSTLC (x : env) b
    In an equation for ‘evalSTLC’:
        evalSTLC env (SLam b) = \ x -> evalSTLC (x : env) b
Failed, modules loaded: none.
~~~~

~~~~ {.haskell}
-- What type do we give this?  What's the return type?

evalSTLC env (SVar v) = lkup v env
evalSTLC env (SNum i) = i
evalSTLC env (SApp f x) = (evalSTLC env f) (evalSTLC env x)
evalSTLC env (SLam b) = \x -> evalSTLC (x : env) b
~~~~

# A Less Simple (But Still Powerful) DSL

~~~~ {.haskell}
data Tag = IntTag Int
         | LambdaTag (Tag -> Tag)

evalSTLC :: [Tag] -> STLCTerm -> Tag
evalSTLC env (SVar v) = lkup v env
evalSTLC _   (SNum i) = IntTag i
evalSTLC env (SApp f x) = go (evalSTLC env f)
  where
    evx = evalSTLC env x
    go (LambdaTag f') = f' evx
    go z             =
      error $ "Can't apply the non-function '" ++ show z ++
      "' to argument '" ++ show evx ++ "'!"
evalSTLC env (SLam b) = LambdaTag $ \x -> evalSTLC (x : env) b

instance Show Tag where
  show (IntTag i) = "<" ++ show i ++ " :: Int>"
  show (LambdaTag _) = "<lambda>"

-- our test program: (λx.x) 22
testSTLC = SApp (SLam (SVar VZero)) (SNum 22)
~~~~

~~~~ {.haskell}
> testSTLC
SApp (SLam (SVar VZero)) (SNum 22)
> evalSTLC [] testSTLC
<22 :: Int>
~~~~

# A Less Simple (But Still Powerful) DSL

 - The interpreter is partial!

~~~~ {.haskell}
failSTLC = SApp (SNum 22) (SNum 33)

> evalSTLC [] failSTLC
*** Exception: Can't apply the non-function '<22 :: Int>' to argument '<33 :: Int>'!
~~~~

# A Less Simple (But Still Powerful) DSL

 - The interpreter is partial!

~~~~ {.haskell}
failSTLC = SApp (SNum 22) (SNum 33)

> evalSTLC [] failSTLC
*** Exception: Can't apply the non-function '<22 :: Int>' to argument '<33 :: Int>'!
~~~~

 - Code is more complicated

 - We have some runtime overhead due to pattern-matching on the tag

# A Less Simple (But Still Powerful) DSL

 Property                     ADTs
---------                     ----------
Multiple Interpretations       :)
Expression Problem             :(
Static Safety                  :(

# A Tagless Encoding

Let's tackle the Static Safety Problem by using Generalised Algebraic
Data Types (GADTs).

~~~~ {.haskell}
{-# LANGUAGE GADTs #-}

data Term t where
  -- This quantified type could be a particular type like `Int`, but
  -- I'm using anything I can `show` just to make interactive
  -- debugging easier
  Const :: Show a => a -> Term a
  Var :: a -> Term a
  App :: Term (b -> a) -> Term b -> Term a
  Lam :: (Term b -> Term a) -> Term (b -> a)

eval :: Term t -> t
eval (Var x) = x
eval (Const x) = x
eval (App f x) = eval f (eval x)
eval (Lam f) = eval . f . Var
~~~~

# A Tagless Encoding

Let's try some test programs.

~~~~ {.haskell}
first = Lam $ \x -> Lam $ \y -> x

twice = Lam $ \f -> Lam $ \x -> Lam $ \y -> f `App` (f `App` x `App` y) `App` y

pairs = twice `App` first `App` (Const 33) `App` (Const 22)
~~~~

~~~~ {.haskell}
> eval pairs
33

> :t pairs
pairs :: (Num a, Show a) => Term a
~~~~

# A Tagless Encoding

~~~~ {.haskell}
failure = twice `App` (Const 33) `App` (Const 22) `App` first
~~~~

~~~~ {.text}
FinallyTagless.hs:210:1:
    Could not deduce (Num (a -> (a0 -> b0 -> a0) -> a))
    from the context (Num a,
                      Num (a -> (a1 -> b -> a1) -> a),
                      Show a,
                      Show (a -> (a1 -> b -> a1) -> a))
      bound by the inferred type for ‘failure’:
                 (Num a, Num (a -> (a1 -> b -> a1) -> a), Show a,
                  Show (a -> (a1 -> b -> a1) -> a)) =>
                 Term a
      at FinallyTagless.hs:210:1-61
    The type variables ‘b0’, ‘a0’ are ambiguous
    When checking that ‘failure’ has the inferred type
      failure :: forall a b a1.
                 (Num a, Num (a -> (a1 -> b -> a1) -> a), Show a,
                  Show (a -> (a1 -> b -> a1) -> a)) =>
                 Term a
    Probable cause: the inferred type is ambiguous
~~~~

# A Tagless Encoding

~~~~ {.haskell}

-- Users want Turing-completeness.  Management has demanded we add a
-- fixpoint combinator without breaking any existing code!

data TermY t where
  ConstY :: Show a => a -> TermY a
  VarY :: a -> TermY a
  AppY :: TermY (b -> a) -> TermY b -> TermY a
  LamY :: (TermY b -> TermY a) -> TermY (b -> a)
  Y    :: TermY (a -> a) -> TermY a

evalY :: TermY t -> t
evalY (VarY x) = x
evalY (ConstY x) = x
evalY (AppY f x) = evalY f (evalY x)
evalY (LamY f) = evalY . f . VarY
evalY (Y f) = y' $ evalY f
  where
    y' g = g $ y' g

twiceY = LamY $ \f -> LamY $ \x -> LamY $ \y -> f `AppY` (f `AppY` x `AppY` y) `AppY` y
~~~~

# A Tagless Encoding

 Property                     ADTs        GADTs
---------                     ----------  --------
Multiple Interpretations       :)          :)
Expression Problem             :(          :(
Static Safety                  :(          :)

# A Final Encoding

~~~~ {.haskell}
newtype Eval a = Eval { eval :: a }

var :: Show a => a -> Eval a
var = Eval

app :: Eval (b -> a) -> Eval b -> Eval a
app (Eval f) (Eval x) = Eval $ f x

lam :: (Eval a -> Eval b) -> Eval (a -> b)
lam b = Eval $ \x -> eval (b (Eval x))
~~~~

# A Final Encoding

~~~~ {.haskell}
-- first = Lam $ \x -> Lam $ \y -> x
first = lam $ \x -> lam $ \y -> x

-- twice = Lam $ \f -> Lam $ \x -> Lam $ \y -> f `App` (f `App` x `App` y) `App` y
twice = lam $ \f -> lam $ \x -> lam $ \y -> f `app` (f `app` x `app` y) `app` y

-- pairs = twice `App` first `App` (Const 33) `App` (Const 22)
pairs = twice `app` first `app` var 33 `app` var 22

> :t pairs
pairs :: (Num a, Show a) => Eval a

> eval pairs
33
~~~~

# A Final Encoding

~~~~ {.haskell}
failure = twice `app` (var 33) `app` (var 22) `app` first

FinallyTagless.hs:72:5:
    Non type-variable argument
      in the constraint: Num (a -> (b -> a -> b) -> a)
    (Use FlexibleContexts to permit this)
    When checking that ‘failure’ has the inferred type
      failure :: forall a a1 b.
                 (Num a, Num (a -> (b -> a1 -> b) -> a), Show a,
                  Show (a -> (b -> a1 -> b) -> a)) =>
                 Eval a
~~~~

# A Final Encoding

~~~~ {.haskell}
import Final.hs (Eval(..), lam, app, var)

y :: Eval (a -> a) -> Eval a
y (Eval f) = Eval (y' f)
  where
    y' g = g $ y' g

twentytwo = y $ lam $ \x -> const (var 22) x
~~~~

~~~~ {.haskell}
> eval $ twentytwo
22
~~~~

# A Final Encoding

 Property                     ADTs        GADTs     Functions
---------                     ----------  --------  ----------
Multiple Interpretations       :)          :)        :(
Expression Problem             :(          :(        :)
Static Safety                  :(          :)        :)

# The Finally Tagless Solution

~~~~ {.haskell}
class FinalTerm repr where
  var :: Show a => a -> repr a
  app :: repr (b -> a) -> repr b -> repr a
  lam :: (repr a -> repr b) -> repr (a -> b)
~~~~

Now we can make an instance for evaluation.

~~~~ {.haskell}
newtype Eval a = Eval { eval :: a }

instance FinalTerm Eval where
  var x = Eval x
  app (Eval f) (Eval x) = Eval $ f x
  lam b = Eval $ \x -> eval' (b (Eval x))
~~~~

# The Finally Tagless Solution

Our test programs are unchanged.

~~~~ {.haskell}
first = lam $ \x -> lam $ \y -> x

twice = lam $ \f -> lam $ \x -> lam $ \y -> f `app` (f `app` x `app` y) `app` y

pairs = twice `app` first `app` var 33 `app` var 22

> :t pairs
pairs :: (Num a, Show a, FinalTerm repr) => repr a

> eval pairs
33
~~~~

# The Finally Tagless Solution

Let's make a second interpreter.

~~~~ {.haskell}
-- This is just machinery for generating variable names
varnames :: [String]
varnames = "xyzwabcdefghmnpqrstu"

parens [] = "()"
parens str
  | head str == '(' && last str == ')' = str
  | otherwise = "(" ++ str ++ ")"

-- Here is the new type we use to choose the instance we want.
newtype Pretty a = Pretty { unPretty :: [String] -> String }

instance FinalTerm Pretty where
  var = Pretty . const . show
  lam f = Pretty $ \(v:vs) -> parens $
    "lambda " ++ v ++ ". " ++ unPretty (f (Pretty $ const v)) vs
  app (Pretty f) (Pretty x) = Pretty $ \c -> parens $ f c ++ " " ++ x c

pretty :: Pretty a -> String
pretty expr = unPretty expr varnames
~~~~

~~~~ {.haskell}
> pretty' pairs
"(((lambda x. (lambda y. (lambda z. ((x ((x y) z)) z)))) (lambda x. (lambda y. x)) 33) 22)"
~~~~

# The Finally Tagless Solution

~~~~ {.haskell}
import FinallyTagless.hs (FinalTerm(..), Eval(..), Pretty(..), pretty)

class FinalTermY repr where
  y :: repr (a -> a) -> repr a

instance FinalTermY Eval where
  y (Eval f) = Eval $ y' f
    where
      y' g = g $ y' g

instance FinalTermY Pretty where
  y (Pretty f) = Pretty $ \names@(v:_) ->
    parens $ "fixpoint of " ++ f names ++ " with respect to '" ++ v ++ "'"

~~~~

~~~~ {.haskell}
twentytwo = y $ lam $ \x -> const (var 22) x

> eval twentytwo
22

> pretty twentytwo
"(fixpoint of (lambda x. 22) with respect to 'x')"
~~~~

# The Finally Tagless Approach

 Property                     ADTs        GADTs     Functions    Type Classes
---------                     ----------  --------  ----------   -------------
Multiple Interpretations       :)          :)        :(           :)
Expression Problem             :(          :(        :)           :)
Static Safety                  :(          :)        :)           :)

# Equivalence Of Final and Initial Encodings

~~~~ {.haskell}
data HTerm h t where
  HVar :: Show t => t -> HTerm h t
  HCell :: h t -> HTerm h t
  HApp :: HTerm h (a -> b) -> HTerm h a -> HTerm h b
  HLam :: (HTerm h a -> HTerm h b) -> HTerm h (a -> b)

-- An interpreter for Eval
evalH :: HTerm Eval t -> t
evalH (HVar x) = x
evalH (HApp f x) = evalH f (evalH x)
evalH (HCell x) = eval' x
evalH (HLam b) = evalH . b . HCell . Eval

-- Now we can convert from final to initial:
instance FinalTerm (HTerm h) where
  var = HVar
  app = HApp
  lam = HLam

-- and from initial to final:
convert :: (FinalTerm repr) => HTerm repr t -> repr t
convert (HVar x) = var x
convert (HCell x) = x
convert (HApp f x) = app (convert f) (convert x)
convert (HLam b) = lam $ convert . b . HCell
~~~~

# Drawbacks

There are some disadvantages . . .

~~~~ {.haskell}
runAndPrettyPrint prog = (eval prog, pretty prog)
~~~~

~~~~ {.haskell}
FinallyTagless.hs:402:42:
    Couldn't match expected type ‘Pretty a0’ with actual type ‘Eval t’
    Relevant bindings include
      t :: Eval t (bound at FinallyTagless.hs:402:20)
      runAndPrettyPrint' :: Eval t -> (t, String)
        (bound at FinallyTagless.hs:402:1)
    In the first argument of ‘pretty'’, namely ‘t’
    In the expression: pretty' t
~~~~

> Polymorphism in Haskell is not first-class. -- Oleg

When we pattern-match on the program, we constrain it to a single
type.  Alas!

# Drawbacks

~~~~ {.haskell}
{-# LANGUAGE RankNTypes #-}

runAndPrettyPrint :: (forall repr. FinalTerm repr => repr t) -> (t, String)
runAndPrettyPrint prog = (eval prog, pretty prog)
~~~~

~~~~ {.haskell}
> runAndPrettyPrint pairs
(33,"(((lambda x. (lambda y. (lambda z. ((x ((x y) z)) z)))) (lambda x. (lambda y. x)) 33) 22)
")
~~~~

# Drawbacks

~~~~ {.haskell}
{-# LANGUAGE RankNTypes #-}

runAndPrettyPrint :: (forall repr. FinalTerm repr => repr t) -> (t, String)
runAndPrettyPrint prog = (eval prog, pretty prog)
~~~~

~~~~ {.haskell}
> runAndPrettyPrint pairs
(33,"(((lambda x. (lambda y. (lambda z. ((x ((x y) z)) z)))) (lambda x. (lambda y. x)) 33) 22)
")
~~~~

~~~~ {.haskell}
> runAndPrettyPrint twentytwo
<interactive>:164:20:
    Could not deduce (FinalTermY repr)
      arising from a use of ‘twentytwo’ . . .
~~~~

 Property                     Type Classes  RankNTypes
---------                     ------------- -------------
Multiple Interpretations       :)            :)
Expression Problem             :)            :(
Static Safety                  :)            :)
Simultaneous interpretations   :(            :)

# Drawbacks

The only thing we can do with a finally-encoded term is to interpret
it.

~~~~ {.haskell}
newtype Pair repr repr' t = Pair { unPair :: (repr t, repr' t) }

instance (FinalTerm repr, FinalTerm repr') => FinalTerm (Pair repr repr') where
  var x = Pair (var x, var x)
  app f x = let (f', f'') = unPair f
                (x', x'') = unPair x
            in Pair (app f' x', app f'' x'')
  lam b = Pair (lam $ \x -> fst . unPair . b $ Pair (x, undefined),
                lam $ \z -> snd . unPair . b $ Pair (undefined, z))
~~~~

~~~~ {.haskell}
runAndPrettyPrint prog = (eval $ l prog, pretty $ r prog)
  where
    l = fst . unPair
    r = snd . unPair
~~~~

 Property                     Type Classes  RankNTypes   Pair instance
---------                     ------------- -----------  -------------
Multiple Interpretations       :)            :)           :)
Expression Problem             :)            :(           :)
Static Safety                  :)            :)           :)
Simultaneous interpretations   :(            :)           :)

# Initial Advantages

Here are some advantages of the ADT method, also known as the
"initial" encoding of the language:

 * Pattern-matching

 * No extra machinery once we accept the need for language extensions

 * Naturally extends to type systems that can't be embedded

# Final Advantages

Here are some advantages of the typeclass method, also known as the
"final" encoding of the language:

 * Solves the Expression Problem in Haskell 98

 * Function composition for substituting expressions

# Other Notes

 * "Coproducts for free": can this initial encoding solve the
   expression problem?

 * Finally Tagless can implement linear/affine LC.

 * De Bruijn works too.

# Conclusion

[Slides](https://peddie.github.io/encodings/encodings.html) and
[annotated
slides](https://peddie.github.io/encodings/encodings-text.html) can be
found at

    https://peddie.github.io/encodings

Finally Tagless can be found at

    http://okmij.org/ftp/tagless-final/JFP.pdf

The (very helpful) course notes can be found at

    http://okmij.org/ftp/tagless-final/course

Wikipedia has articles on

 * GADTs
 * Church encodings
 * Scott encodings

# Backups

# De Bruijn Indices, Finally Encoded

~~~~ {.haskell}
class FinalDBTerm repr where
  dbvar :: Show a => a -> repr env a
  dbapp :: repr env (a -> b) -> repr env a -> repr env b
  -- The type of 'dblam' is pretty different from the type of the HOAS
  -- method 'lam'.  If you think about how 'EvalEnv' is defined, it
  -- makes sense -- the interpreter is going to have to wrap a
  -- function and get the argument from the environment.  In the HOAS
  -- case, we simply make the function explicit in the type of the
  -- method.
  dblam :: repr (a, env) b -> repr env (a -> b)

  -- This method states that the environment contains a value of the
  -- correct type to be applied: lambda 0
  zero :: repr (a, env) a
  -- This method states that the environment is being extended with a
  -- value of type 'b'.
  succ :: repr env a -> repr (b, env) a

class FinalDBTermArith repr where
  dbadd :: Num a => repr env a -> repr env a -> repr env a
  dbmul :: Num a => repr env a -> repr env a -> repr env a
  dbneg :: Num a => repr env a -> repr env a

newtype EvalEnv env a = EvalEnv { evalEnv :: env -> a }

instance FinalDBTerm EvalEnv where
  dbvar x = EvalEnv (const x)
  dbapp (EvalEnv f) (EvalEnv x) = EvalEnv $ \env -> (f env) (x env)
  dblam (EvalEnv body) = EvalEnv $ \env x -> body (x, env)

  zero = EvalEnv fst
  succ (EvalEnv v) = EvalEnv $ v . snd

instance FinalDBTermArith EvalEnv where
  dbadd (EvalEnv a) (EvalEnv b) = EvalEnv $ \env -> a env + b env
  dbmul (EvalEnv a) (EvalEnv b) = EvalEnv $ \env -> a env * b env
  dbneg (EvalEnv a) = EvalEnv $ \env -> negate $ a env

dbeval expr = evalEnv expr ()

dbtest' = dblam (dbadd (succ zero) (succ (succ zero)))

dbtest'' = ((dblam (dblam dbtest') `dbapp` dbvar 22) `dbapp` dbvar (33 :: Double)) `dbapp` undefined

dbtest = dblam (dbadd zero zero) `dbapp` dbvar (22 :: Double)
~~~~

# De Bruijn Indices, Initially Encoded

~~~~ {.haskell}
data DBVar env t where
  DBZ :: DBVar (t, env) t
  DBS :: DBVar env t -> DBVar (a, env) t

data DBTerm env t where
  DBConst :: t -> DBTerm env t
  DBVar :: DBVar env t -> DBTerm env t
  DBLam :: DBTerm (a, env) b -> DBTerm env (a -> b)
  DBApp :: DBTerm env (a -> b) -> DBTerm env a -> DBTerm env b

dbLookup :: DBVar env t -> env -> t
dbLookup  DBZ    (x, _)   = x
dbLookup (DBS v) (_, env) = dbLookup v env

dbEval :: env -> DBTerm env t -> t
dbEval env (DBVar v) = dbLookup v env
dbEval env (DBConst t) = t
dbEval env (DBLam body) = \x -> dbEval (x, env) body
dbEval env (DBApp f arg) = (dbEval env f) (dbEval env arg)

dbinitialtest = DBApp (DBLam (DBVar DBZ)) (DBConst 22)
~~~~

# De Bruijn Index to HOAS Conversion

For this final trick, I don't think there's a way to get away without
either multi-parameter type classes with functional dependencies or
type families.  Our environment needs to have the same tuple
structure, but all the types need to change from e.g. `a` to `repr a`
(though the empty environment is still simply `()`).  We don't have a
general way to express this with Haskell 98.  Type families (or MPTCs
+ fundeps) let us write type-level functions to explain to the type
checker the appropriate relationships between the types in the
environment.

Note that the conversion remains open!  We could add new De
Bruijn-based classes and new HOAS-based counterparts in a new module,
along with the translation instances, and reuse this original `toHOAS`
function!

~~~~ {.haskell}
type family Env repr a where
  Env repr () = ()
  Env repr (a, env) = (repr a, Env repr env)

newtype WrapHOAS repr env a = WrapHOAS { unwrapHOAS :: Env repr env -> repr a }

instance FinalTerm repr => FinalDBTerm (WrapHOAS repr) where
  dbvar x = WrapHOAS (const $ var x)
  dbapp (WrapHOAS f) (WrapHOAS x) = WrapHOAS $ \env -> app (f env) (x env)
  dblam (WrapHOAS body) = WrapHOAS $ \env -> lam $ \x -> body (x, env)

  zero = WrapHOAS fst
  succ (WrapHOAS v) = WrapHOAS $ v . snd

instance FinalTermArith repr => FinalDBTermArith (WrapHOAS repr) where
  dbadd (WrapHOAS a) (WrapHOAS b) = WrapHOAS $ \env -> a env `add` b env
  dbmul (WrapHOAS a) (WrapHOAS b) = WrapHOAS $ \env -> a env `mul` b env
  dbneg (WrapHOAS a)              = WrapHOAS $ \env -> neg $ a env

toHOAS :: WrapHOAS repr () a -> repr a
toHOAS expr = unwrapHOAS expr ()
~~~~

# Scott Encodings

~~~~ {.haskell}
{-# LANGUAGE RankNTypes #-}
newtype SList a = SL { elim :: forall b. (a -> SList a -> b) -> b -> b }

snil :: forall a. SList a
snil = SL $ \c n -> n

scons :: forall a. a -> SList a -> SList a
scons x xs = SL $ \c n -> c x xs

smap :: forall a b. (a -> b) -> SList a -> SList b
smap f lst = elim lst (\x xs -> f x `scons` smap f xs) snil
~~~~

# Drawbacks

~~~~ {.haskell}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ImpredicativeTypes #-}

newtype Program a = Program {
  getProgram :: forall repr. FinalTerm repr => repr a
  }

convert' :: HTerm (forall repr. FinalTerm repr => repr :: * -> *) t
         -> (forall repr. FinalTerm repr => repr t)
convert' (HVar x) = var x
convert' (HCell x) = x
convert' (HApp f x) = app (convert' f) (convert' x)
convert' (HLam b) = lam $ convert' . b . HCell
~~~~

If we used the ImpredicativeTypes extension, which lets us write
things like this, we could solve the problem.  Sadly, GHC would no
longer be able to infer types for such values.  That's a heavy price
to pay!
