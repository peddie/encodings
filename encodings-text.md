% Initial and Final Encodings

# Introduction

Functional programming is all about composition.  I'd like to show you
one way that simple function composition can be very powerful in
combination with a really interesting principle about how we represent
data in our programs.

One of the big ideas that functional programming (Lisp!) introduced
back in the day is the idea that functions are just data like anything
else, and that conversely, data can often be represented as functions.
If you've done programming work in any flavor of Lisp, you've most
likely run into ad-hoc applications of this principle (the
metaprogramming system relies on it, but that looks a bit different).

It turns out that within programming language and type theory, this
principle can be made more formal and universal.  Happily, knowing
when to look at either side of the data/function coin can have some
concrete practical implications.  Today I'd like to explain what I've
understood about this principle and what it means in terms of
practical programming.  We'll look at some simple examples to
illustrate the principle, and then I'm planning to spend the majority
of the talk showing how this all looks when you apply it to embedded
domain specific languages, mostly following the paper "Finally
Tagless, Partially Evaluated" by Jacques Carette, Oleg Kiselyov and
Chung-chieh Shan.  (Some of you may have heard of this paper already.)

I'm going to present this stuff using Haskell, because I am most
comfortable with it (and I suspect a lot of BFPGers can understand it
well), but very little is Haskell-specific, and in fact the paper is
mostly in OCaml.

One final caveat: what I'm going to present is from the Finally
Tagless paper, plus additional material and understanding gained from
coding it all up, reading stackoverflow/reddit posts and BSing with
Matt B. and Dave.  I have absolutely no formal background in this
stuff, and I suspect there's a lot more theory behind it that explains
it better.  I wanted to give the talk because without any of that, I
was able to use the technique to get some concrete, practical benefits
when building a real application, and I wanted to spread the cool
ideas in case it helps.  You don't have to know a lot of deep theory
to put this to work!

Please just shout out questions if I move on and something is still
confusing; you're probably not the only one.

# The Basic Idea

Consider the humble linked list.

~~~~ {.haskell}
data List a = Nil
            | Cons a (List a)
~~~~

What can you do with a list?  The basic functionality is that you can
do something

 - starting at the head
 - in order

This could be a `reverse`, a `fold` or whatever.

# The Basic Idea

TODO(MP): DLists instead?

TODO(MP): Show step-by-step evaluation to explain this

What if we turned the list inside-out?

~~~~ {.haskell}
{-# LANGUAGE RankNTypes #-}  -- so we can use 'forall'

newtype Church a = Church {
  runChurch :: forall r. (a -> r -> r)    -> r   -> r
               --          cons           nil    result
  }
~~~~

This is called a Church encoding, and it represents a `List` as a
higher-order function.  This function takes two arguments, one which
it can call in the case of a `Cons` and the other of which it can use
in the case of a `Nil`.  To make this clearer, let's see how to go
back and forth:

~~~~ {.haskell}
convert :: [a] -> Church a
convert lst = Church $ \c n -> foldr c n lst
~~~~

Converting from an ordinary linked list to a Church-encoded list means
generating a function that, when run with the arguments `c` and `n`,
will let the caller perform the same operations.  It's easiest for me
to puzzle it out by looking at the types.

~~~~ {.haskell}
foldr :: Foldable t => (a -> b -> b) -> b -> t a ->
c :: a -> r -> r  -- the folding function
n :: r            -- the terminating value
~~~~

The list is expressed as a fold across the real list, and at each
step, it applies the given `c` to both the value from `lst` and `n`.
To do operations on this list, you must provide a function that will
run on the piece of data in the cons cell (the first `a` in `Cons a
(List a)`) and the remainder of the list, and a value to be passed in
if the list ends (with a `Nil`).  The list has been rearranged from a
plain old data structure to a higher-order function that captures the
same computation structure.  Here is how to go in the other direction:

~~~~ {.haskell}
excommunicate :: Church a -> [a]
excommunicate ch = runChurch ch (:) []
~~~~

I think this is pretty straightforward if you've understood the
function to go in the first direction.  `ch` is the higher-order
function that will walk our arguments through the structure of the
list.  The argument `(:)` says what we want to do every time we are in
a `Cons` situation.  The argument `[]` says what we want to do if we
hit a `Nil`.

# The Basic Idea

Let's look at one more function, `append` (`++` with ordinary lists):

~~~~ {.haskell}
appendChurch :: Church a -> Church a -> Church a
appendChurch xs ys = Church $ \c n -> runChurch xs c $ runChurch ys c n
~~~~

What we're doing here is saying "OK, when you run the first list, when
you get to the end of the list and encounter `Nil`, just carry on into
the result of running the second list."  In other words, we can put
the two lists together end-to-end by using higher-order functions.
You can see that this is just composition if I make the code a bit
more pointless:

~~~~ {.haskell}
appendChurch' :: Church a -> Church a -> Church a
appendChurch' (Church fx) (Church fy) = Church $ \c -> fx c . fy c
~~~~

# The Basic Idea

But why on earth would you want to deal with this convoluted
representation?  Consider the asymptotics:

~~~~ {.haskell}
(++) :: [a] -> [a] -> [a]
[]     ++ bs = bs
(a:as) ++ bs = a : (as ++ bs)

-- call `:` once per element of the first list, every time you call `++`!
~~~~

~~~~ {.haskell}
appendChurch' :: Church a -> Church a -> Church a
appendChurch' (Church fx) (Church fy) = Church $ \c -> fx c . fy c

-- run one function composition per call to `appendChurch'`!
~~~~

Scanning through `as` step-by-step and rebuilding it with `bs` at the
end takes time proportional to the length of `as` (though it doesn't
matter how long `bs` is).  This is fine if you're appending once, but
if you're cruising along and tacking things on to the end of the list
repeatedly, this starts to get slow (it's quadratic in `length as`).
One common situation where this might happen is if you are logging
something (e.g. with `Writer`).

If we know that function composition is constant-time, then we can see
how with the church-encoded list, the time required to append one list
to another doesn't depend on either list's length, so you won't pay
the extra cost as `as` gets longer and longer.  If you've seen things
like continuation-passing style transformations, codensity
transformation for free monads or "Free Monads for Less", this is a
similar idea.  I'm not going to say anything else about those, but we
can talk afterwards.

Church encodings can also solve other practical problems.  Witness the
following IRC interaction from 4 November on `#bfpg`:

> 04:38:09          bkolera | It wouldn't be a big deal if
>                           | we didn't encode sum types as
>                           | sub types in scala, but ...
>                           | >_>
> 04:42:56          georgew | You could just Church encode?

The motto rears its head again: "it's all about composition!"  Using
composition and representing our data as code lets us get an
asymptotic speedup on a very common usage pattern.

# Embedded Domain-Specific Languages

Now that we've seen a simple example of what representing our data as
functions might mean and how it might help us, let's think about a
more general case.  One common approach to problem solving in
functional programming is to define a small **domain-specific
language** (DSL) in which to represent the problem or class of
problems we're interested in attacking.  Calling a DSL **embedded**
means that rather than have a separate parser, typechecker, etc. for
our new language, we're going to just write regular programs with
whatever language we like, but ideally make it look like and work like
a language on its own.  EDSL programs are just regular programs.  We
can then provide a way to run programs written in this so-called
"object language" in our "meta language" (e.g. Haskell or OCaml, or
some very general purpose language with a great type system and
compiler to do the heavy lifting).  This approach can help decouple
the problem domain from the engineering required to make it happen
cleanly and efficiently on the computer.  Let's look at some common
considerations when implementing a domain-specific language.

# The Multiple Interpretations Problem

A common design goal when implementing a DSL is that you'd like to be
able to interpret the same program in different ways.  The obvious way
is:

  * Run the program and get its result

But there are other ways, for example:

  * Pretty-print or serialize the program

  * Perform optimizations before executing the program

  * Run additional checks or pre-run tests (particularly if the
    language allows side effects outside the type system).

  * Generate code, e.g. spit out C or Javascript so our program can
    be run in restricted environments.

  * Execute using alternative evaluation strategies, e.g. using the
    LLVM JIT compilation tools to speed up execution or running the
    program on a GPU.

# The Expression Problem

> The Expression Problem is a new name for an old problem.  The goal
> is to define a datatype by cases, where one can add new cases to the
> datatype and new functions over the datatype, without recompiling
> existing code, and while retaining static type safety (e.g., no
> casts).

  -- Philip Wadler, Nov. 1998

In other words, we'd like to be able to extend a DSL without having to
modify its core.  This can allow users to extend the language
themselves or to provide new features without breaking existing
systems.

# The Static Safety Problem

A common design goal when implementing an embedded DSL is that we'd
like to take advantage of our metalanguage's facilities where we can.
In particular, if we're building a fairly simple language, it would be
very convenient to be able to lean on the sophisticated type checker
of the metalanguage at compile time.  A downside is that the error
messages may be clearer to the implementer than the user, but the
upside is that we can catch errors in the DSL at compile time with a
lot less work than if we implemented our own type checker.  The same
kinds of advantages can apply for debugging tools, code generation,
etc.

# Solving All The Problems At Once

I'm now going to introduce the technique from "Finally Tagless," which
solves these problems quite nicely.  Later, we'll also discuss some
other advantages and disadvantages to the different representations.
We're going to start with some code that exhibits some of the problems
and work our way towards the solution.  To begin, let's look at our
data in the ordinary way.

# A Simple (But Powerful) DSL

To illustrate how representing data differently can be applied to
building DSLs, let's take the simply-typed lambda calculus, with
integers as the base type, as a running example.  The STLC is a good
example because it has first-class functions, so we'll make sure our
results apply to higher-order languages.

Unfortunately I must use de Bruijn indices, as I haven't figured out
how to use higher-order abstract syntax with plain ADTs.  The later
examples will involve less binding-related machinery.

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
lkup VZero [x] = x
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

This can't work; the resulting expression might be either an `SLam` or
an `SNum`.  We can't put both into our environment or return both!  So
we have to introduce a "tag" to wrap the possible values and indicate
which we're dealing with.

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

This code typechecks and runs.

~~~~ {.haskell}
> testSTLC
SApp (SLam (SVar VZero)) (SNum 22)
> evalSTLC [] testSTLC
<22 :: Int>
~~~~

# A Less Simple (But Still Powerful) DSL

Sadly, we have to introduce this tag to get things to typecheck.  The
interpreter is less clear and less direct.  In addition

 - We have some runtime overhead due to pattern-matching on the tag

 - The interpreter is partial!

~~~~ {.haskell}
failSTLC = SApp (SNum 22) (SNum 33)

> evalSTLC [] failSTLC
*** Exception: Can't apply the non-function '<22 :: Int>' to argument '<33 :: Int>'!
~~~~

This term is nonsensical, but it compiled just fine.

# A Less Simple (But Still Powerful) DSL

This interpreter solves the Multiple Interpretations Problem.  We can
take an `STLCTerm` and pass it around, manipulate it, optimize it,
pretty-print it, etc.

This interpreter does not solve the Expression Problem.  To extend the
language, we have to open up this file, change the definition of
`STLCTerm` and rewrite a bunch of code to match.  Our compiler should
help highlight what we need to update, but it's still not
satisfactory.

This interpreter does not solve the Static Safety Problem.  The
language is extremely simple, but our implementation doesn't let us
lean on Haskell's type checker.  We have to write our own.

# A Tagless Encoding

Let's tackle the Static Safety Problem by using Generalised Algebraic
Data Types (GADTs).

~~~~ {.haskell}
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

Well, that was easy.  Let's try some test programs.

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

I'm not sure it was worth computing, but it worked fine.

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

It looks like we've licked the static safety problem!

This interpreter also solves the Multiple Interpretations Problem
(just write any function that pattern-matches on the constructors of
`Term`).

# A Tagless Encoding

Unfortunately, this GADT-based interpreter is the same as the previous
interpreter with respect to the Expression Problem.

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

Ugh, we've got to rewrite everything!  Management is disappointed.

At this point, I hope it's clear that to solve the Expression Problem,
we can't simply use ordinary data types with constructors directly.
We'll never be able to extend them in a separate module.  Let's try a
different approach, motivated by our earlier look at Church encoding.

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

We can do a similar trick to the Church encoding of lists and
represent our data constructors as functions.  This is **not** a
Church encoding; it's called a **final encoding** for reasons unknown
to me.  The version using explicit constructors is called an **initial
encoding**.

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

The new encoding works, and it's quite similar to the original!  Let's
see how we do on the various problems.

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

Lovely!  This final encoding solves the Static Safety Problem.

# A Final Encoding

Now we can extend our object language in a different module, without
touching the original.

~~~~ {.haskell}
import Final.hs (Eval(..), lam, app, var)

y :: Eval (a -> a) -> Eval a
y (Eval f) = Eval (y' f)
  where
    y' g = g $ y' g

twentytwo = y $ lam $ \x -> const (var 22) x
~~~~

Let's try it out:

~~~~ {.haskell}
> eval $ twentytwo
22
~~~~

Hooray!  This interpreter solves the Expression Problem!  Management
is satisfied!  But . . .

# A Final Encoding

How do we solve the Multiple Interpretations Problem?  There's nothing
we can change about the way we interpret our code.  Optimization is
right out!  This leaves our users unhappy again.

We solved the Expression Problem by representing our data as
functions, but now we want our functions to have different meanings.
Can you think of a way to give functions different meanings, depending
on some context?  Depending on the type?

# The Finally Tagless Solution

The answer, in Haskell, is to use type classes.  (In OCaml, you can do
the same thing with a module.)  This type class defines the lambda
calculus language:

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

We can also make a second interpreter: a pretty-printer!

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

Let's run the test program through it:

~~~~ {.haskell}
> pretty' pairs
"(((lambda x. (lambda y. (lambda z. ((x ((x y) z)) z)))) (lambda x. (lambda y. x)) 33) 22)"
~~~~

# The Finally Tagless Solution

We've already shown that we can solve the Multiple Interpretations
Problem, and the Static Safety Problem is solved just as with
functions.  Now let's see how this approach addresses the Expression
Problem.

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

This is not an insubstantial addition.

~~~~ {.haskell}
twentytwo = y $ lam $ \x -> const (var 22) x

> eval twentytwo
22

> pretty twentytwo
"(fixpoint of (lambda x. 22) with respect to 'x')"
~~~~

You may have noticed that in these cases, type inference also works
perfectly fine, and we haven't used anything outside Haskell 98.

# Equivalence Of Final and Initial Encodings

We can convert back and forth between encodings to show that they are
equivalent.  There is a catch, though; if we used our old initial
encoding (the `Term` GADT), we'd have to commit to a particular final
interpretation, e.g. `Eval` or `Pretty`.  Instead, I've defined this
new initial encoding containing the `HCell` constructor.  This
constructor lets us work with terms that have already been "finally
encoded" in our initial representation.  It doesn't show up in any
programs, but it's necessary to write `evalH`.

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

Notice that when you convert to or from a particular initial encoding,
your language becomes 'closed' again.  The initial encoding for the
language and the conversion code above would have to be modified if we
wanted to convert terms using the Y combinator.

# Drawbacks

The Finally Tagless approach to EDSL implementation solves the Static
Safety Problem, the Multiple Interpreters Problem and the Expression
Problem all at once.  Management is happy; users are happy; we're
happy.  Is it the end of the talk?

There are some disadvantages . . .

~~~~ {.haskell}
runAndPrettyPrint prog = (eval' prog, pretty' prog)
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

What is this about?!  Couldn't match `Pretty a0` with `Eval t`?

This problem occurs with any function that produces or accepts a
finally-encoded program; Oleg describes this by saying that
"polymorphism in Haskell is not first-class" -- when we pattern-match
on the program, we constrain it to a single type.  Alas!

We can use this `Program` wrapper if we want to output a program in
our object language.  This achieves the goal of allowing multiple
interpretations, but at a price: the representation is now closed,
since we had to explicitly list what parts of the language we want to
quantify over.  In this example, we have excluded the fixpoint
combinator from the language!

In addition, wrapping up object language programs with a `forall` also
causes issues for the isomorphism function above (I think); I can't
work out how to write a function that takes an `HTerm repr t` and
converts it to a `Program t`.  We need the `repr` argument to `HTerm`
to be quantified the same way as the one hidden inside the `Program`.

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

# Drawbacks

Remember, the only thing we can do with a finally-encoded term is to
interpret it.  So if we want to use it for multiple things, can we
make an interpreter that provides multiple instances of the program?
This is indeed Oleg's solution to the problem.

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

Equipped with this somewhat clunky tool, we can now build
`runAndPrettyPrint`.

~~~~ {.haskell}
runAndPrettyPrint prog = (eval' $ l prog, pretty' $ r prog)
  where
    l = fst . unPair
    r = snd . unPair
~~~~

We're still in Haskell 98 land, and type inference still works
properly!

# Other Drawbacks

The "Finally Tagless" paper shows how to apply the approach to various
other useful tasks, like deserialization of the program (which
includes type-checking), performing optimizations by transforming the
AST, type-directed partial evaluation, and more.  Those things are
possible (and TDPE is pretty neat), but frankly, it would be a slog to
just show how most of them work (I have backup slides if you're
interested).  For the sake of time and not just blasting more code on
the screen, I'm going to summarize the tradeoffs as I understand them
here.

# Initial Advantages

Here are some advantages of the ADT method, also known as the
"initial" encoding of the language:

* We can pattern-match on the ADT, which makes transforming the
  representation of the embedded program straightforward to understand
  (e.g. if we want to do some optimizations).  It's also amenable to
  existing generic programming tools like 'uniplate' or 'syb'.

* Defining a "tagless" ADT usually requires additional type system
  features, e.g. GADTs, but once the ADT is defined and we accept that
  we need an enhanced type system, object language programs are
  automatically first-class without any additional machinery.  (We
  don't need the weird "duplicating interpreter".)

* If you want to make your initially-encoded language no longer
  embedded directly in the meta language (e.g. moving to a
  dependently-typed language, with a separate type checker), the
  extension is fairly natural.  I haven't seen or come up with any
  clean ways of doing this for finally-encoded ones.

# Final Advantages

Here are some advantages of the typeclass method, also known as the
"final" encoding of the language:

 * Our language is extensible and open; we can add new features to its
   abstract syntax while reusing, without modifying (or even
   recompiling!) the code for the core language.  GHC can infer the
   object program types at the use site.  It's possible to solve the
   Expression Problem.

 * With some usage patterns, especially the typical "monadic
   substitution" pattern that happens when language fragments are
   composed together, a final encoding (with or without type classes)
   can actually yield an asymptotic speedup in creating the program.

 * It is possible to solve the Static Safety Problem in Haskell 98.

# Other Notes

 * I've tried to attack the same three EDSL design problems using some
   of the "coproducts for free" machinery that Dave has written about.
   Unfortunately, I'm currently stumped trying to apply it to
   higher-order languages, and it seems to require an awful lot of
   machinery.  Maybe Dave can do a follow-up talk explaining how this
   can be done nicely!

 * It's worth pointing out that Oleg and others have constructed DSLs
   with linear typing using this approach.  This is noteworthy because
   linear types are not a strict subset of Haskell's or OCaml's type
   system.

 * I've used HOAS for all the examples, but the final approach using
   type classes can be done perfectly fine with de Bruijn indices as
   well.  You can convert from de Bruijn to HOAS pretty
   straightforwardly.  I think the other direction is a lot harder, if
   it's possible.

# Conclusion

If you haven't seen this material before, I hope this talk has
introduced a different perspective on programs and domain-specific
languages.

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
structure, but all the types need to change from e.g. 'a' to 'repr a'
(though the empty environment is still simply '()').  We don't have a
general way to express this with Haskell 98.  Type families (or MPTCs
+ fundeps) let us write type-level functions to explain to the type
checker the appropriate relationships between the types in the
environment.

Note that the conversion remains open!  We could add new De
Bruijn-based classes and new HOAS-based counterparts in a new module,
along with the translation instances, and reuse this original 'toHOAS'
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
