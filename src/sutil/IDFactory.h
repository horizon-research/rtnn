#pragma once

/** Factory for unique ids.
  */
class IDFactory
{
private:
	/** Current id */
	static int id;

public:
	static int getId() { return id++; }
};

